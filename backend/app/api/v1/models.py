"""
Generic Model Management API Endpoints - Comprehensive LLM model configuration and management.
Supports any LLM model through flexible configuration without code changes.
"""

from fastapi import APIRouter, HTTPException, Query, Path, BackgroundTasks, Depends, status
from fastapi.security import HTTPBearer
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, ValidationError
import json
import uuid
from datetime import datetime

from app.models.llm_config import (
    LLMConfig, LLMConfigCreate, LLMConfigUpdate, LLMConfigInfo, LLMConfigPreset,
    LLMConfigTest, LLMConfigTestResult, LLMConfigImport, LLMConfigExport,
    LLMCapabilityDetection, LLMCapabilityDetectionResult, PROVIDER_PRESETS,
    LLMConfigBackup, AuthenticationType, BillingModel, LLMConfigStatus
)
from app.services.llm_service import universal_llm_service
from app.core.deps import get_current_user

router = APIRouter()
security = HTTPBearer()


class PaginationParams(BaseModel):
    """Pagination parameters."""
    page: int = Query(1, ge=1, description="Page number")
    page_size: int = Query(20, ge=1, le=100, description="Items per page")


class FilterParams(BaseModel):
    """Filtering parameters."""
    provider_name: Optional[str] = Query(None, description="Filter by provider name")
    status: Optional[LLMConfigStatus] = Query(None, description="Filter by status")
    search: Optional[str] = Query(None, description="Search in name or description")
    tags: Optional[List[str]] = Query(None, description="Filter by tags")


class SortParams(BaseModel):
    """Sorting parameters."""
    sort_by: str = Query("created_at", description="Sort field")
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order")


# In a real implementation, these would be database operations
# For now, we'll use in-memory storage as demonstration
_model_configs: Dict[str, LLMConfig] = {}
_config_backups: Dict[str, LLMConfigBackup] = {}


async def _log_config_operation(operation: str, config_id: str, user_id: Optional[str] = None):
    """Log configuration operations for audit purposes."""
    # In a real implementation, this would write to a database or logging service
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "operation": operation,
        "config_id": config_id,
        "user_id": user_id
    }
    # For now, just log to console
    print(f"CONFIG AUDIT: {json.dumps(log_entry)}")


@router.post("/register", 
    summary="Register New LLM Model Configuration",
    description="Add a new LLM model configuration with comprehensive settings",
    response_model=Dict[str, Any],
    status_code=status.HTTP_201_CREATED)
async def register_model_config(
    config: LLMConfigCreate,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user)
) -> Dict[str, Any]:
    """Register a new LLM model configuration."""
    try:
        # Generate unique ID
        config_id = str(uuid.uuid4())
        
        # Create full configuration
        full_config = LLMConfig(
            id=config_id,
            **config.dict()
        )
        
        # Test configuration if requested
        test_result = None
        if config.auto_test_on_create:
            try:
                test_result = await universal_llm_service.test_configuration(full_config)
                full_config.is_valid = test_result.success
                
                if not test_result.success:
                    full_config.validation_errors = [test_result.error_message or "Configuration test failed"]
                    full_config.status = LLMConfigStatus.ERROR
            
            except Exception as e:
                full_config.is_valid = False
                full_config.validation_errors = [f"Configuration test error: {str(e)}"]
                full_config.status = LLMConfigStatus.ERROR
        
        # Store configuration
        _model_configs[config_id] = full_config
        
        # Register with service if valid and active
        if full_config.is_valid and full_config.status == LLMConfigStatus.ACTIVE:
            universal_llm_service.register_config(full_config)
        
        # Log configuration creation
        background_tasks.add_task(
            _log_config_operation, 
            "create", 
            config_id, 
            current_user.get("id") if current_user else None
        )
        
        return {
            "message": "Model configuration registered successfully",
            "config_id": config_id,
            "is_valid": full_config.is_valid,
            "status": full_config.status.value,
            "test_result": test_result.dict() if test_result else None,
            "validation_errors": full_config.validation_errors
        }
    
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Configuration validation failed: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register configuration: {str(e)}"
        )


@router.get("/",
    summary="List All Model Configurations",
    description="Get paginated list of model configurations with filtering and sorting",
    response_model=Dict[str, Any])
async def list_model_configs(
    pagination: PaginationParams = Depends(),
    filters: FilterParams = Depends(),
    sort: SortParams = Depends(),
    current_user = Depends(get_current_user)
) -> Dict[str, Any]:
    """List all model configurations with filtering, pagination, and sorting."""
    try:
        # Get all configs
        all_configs = list(_model_configs.values())
        
        # Apply filters
        filtered_configs = all_configs
        
        if filters.provider_name:
            filtered_configs = [c for c in filtered_configs if c.provider_name.lower() == filters.provider_name.lower()]
        
        if filters.status:
            filtered_configs = [c for c in filtered_configs if c.status == filters.status]
        
        if filters.search:
            search_term = filters.search.lower()
            filtered_configs = [
                c for c in filtered_configs 
                if search_term in c.name.lower() or 
                   search_term in (c.description or "").lower()
            ]
        
        if filters.tags:
            filtered_configs = [
                c for c in filtered_configs 
                if any(tag in c.tags for tag in filters.tags)
            ]
        
        # Apply sorting
        reverse = sort.sort_order == "desc"
        if sort.sort_by == "created_at":
            filtered_configs.sort(key=lambda x: x.created_at, reverse=reverse)
        elif sort.sort_by == "name":
            filtered_configs.sort(key=lambda x: x.name.lower(), reverse=reverse)
        elif sort.sort_by == "success_rate":
            filtered_configs.sort(key=lambda x: x.success_rate, reverse=reverse)
        elif sort.sort_by == "total_requests":
            filtered_configs.sort(key=lambda x: x.total_requests, reverse=reverse)
        
        # Apply pagination
        total = len(filtered_configs)
        start_idx = (pagination.page - 1) * pagination.page_size
        end_idx = start_idx + pagination.page_size
        paginated_configs = filtered_configs[start_idx:end_idx]
        
        # Convert to info format
        config_infos = [
            LLMConfigInfo(
                id=config.id,
                name=config.name,
                provider_name=config.provider_name,
                model_name=config.model_name,
                status=config.status,
                created_at=config.created_at,
                success_rate=config.success_rate,
                total_requests=config.total_requests
            ) for config in paginated_configs
        ]
        
        return {
            "configurations": [info.dict() for info in config_infos],
            "pagination": {
                "page": pagination.page,
                "page_size": pagination.page_size,
                "total": total,
                "pages": (total + pagination.page_size - 1) // pagination.page_size
            },
            "filters_applied": filters.dict(exclude_none=True),
            "sort": sort.dict()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list configurations: {str(e)}"
        )


@router.get("/{config_id}",
    summary="Get Model Configuration Details",
    description="Retrieve detailed information for a specific model configuration",
    response_model=Dict[str, Any])
async def get_model_config(
    config_id: str = Path(..., description="Configuration ID"),
    current_user = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get detailed configuration information."""
    if config_id not in _model_configs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration {config_id} not found"
        )
    
    config = _model_configs[config_id]
    
    # Get performance metrics from service
    performance_metrics = await universal_llm_service.get_performance_metrics(config_id)
    
    config_dict = config.dict()
    config_dict["performance_metrics"] = performance_metrics
    
    return {
        "configuration": config_dict,
        "retrieved_at": datetime.utcnow().isoformat()
    }


@router.put("/{config_id}",
    summary="Update Model Configuration",
    description="Update an existing model configuration with partial or complete data",
    response_model=Dict[str, Any])
async def update_model_config(
    config_update: LLMConfigUpdate,
    background_tasks: BackgroundTasks,
    config_id: str = Path(..., description="Configuration ID"),
    current_user = Depends(get_current_user)
) -> Dict[str, Any]:
    """Update an existing model configuration."""
    if config_id not in _model_configs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration {config_id} not found"
        )
    
    try:
        existing_config = _model_configs[config_id]
        
        # Apply updates
        update_data = config_update.dict(exclude_none=True)
        
        for field, value in update_data.items():
            if hasattr(existing_config, field):
                setattr(existing_config, field, value)
        
        existing_config.updated_at = datetime.utcnow()
        
        # Re-register with service if status changed to active
        if (config_update.status == LLMConfigStatus.ACTIVE and 
            existing_config.is_valid):
            universal_llm_service.register_config(existing_config)
        elif config_update.status in [LLMConfigStatus.INACTIVE, LLMConfigStatus.ERROR]:
            universal_llm_service.unregister_config(config_id)
        
        # Log update
        background_tasks.add_task(
            _log_config_operation, 
            "update", 
            config_id, 
            current_user.get("id") if current_user else None
        )
        
        return {
            "message": "Configuration updated successfully",
            "config_id": config_id,
            "updated_fields": list(update_data.keys()),
            "status": existing_config.status.value
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update configuration: {str(e)}"
        )


@router.delete("/{config_id}",
    summary="Delete Model Configuration",
    description="Remove a model configuration from the system",
    response_model=Dict[str, Any])
async def delete_model_config(
    background_tasks: BackgroundTasks,
    config_id: str = Path(..., description="Configuration ID"),
    current_user = Depends(get_current_user)
) -> Dict[str, Any]:
    """Delete a model configuration."""
    if config_id not in _model_configs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration {config_id} not found"
        )
    
    try:
        # Unregister from service
        universal_llm_service.unregister_config(config_id)
        
        # Remove from storage
        deleted_config = _model_configs.pop(config_id)
        
        # Log deletion
        background_tasks.add_task(
            _log_config_operation, 
            "delete", 
            config_id, 
            current_user.get("id") if current_user else None
        )
        
        return {
            "message": "Configuration deleted successfully",
            "config_id": config_id,
            "deleted_config_name": deleted_config.name
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete configuration: {str(e)}"
        )


@router.post("/{config_id}/test",
    summary="Test Model Configuration",
    description="Test connectivity and functionality of a model configuration",
    response_model=LLMConfigTestResult)
async def test_model_config(
    test_request: LLMConfigTest,
    config_id: str = Path(..., description="Configuration ID"),
    current_user = Depends(get_current_user)
) -> LLMConfigTestResult:
    """Test a model configuration."""
    # Use provided config or get from storage
    if test_request.config:
        # Test provided configuration
        temp_config = LLMConfig(id="temp", **test_request.config.dict())
        result = await universal_llm_service.test_configuration(temp_config)
    else:
        # Test stored configuration
        if config_id not in _model_configs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Configuration {config_id} not found"
            )
        
        config = _model_configs[config_id]
        result = await universal_llm_service.test_configuration(config)
        
        # Update stored config with test results
        config.last_health_check = datetime.utcnow()
        if result.success:
            config.success_rate = min(config.success_rate + 0.1, 1.0)
        else:
            config.success_rate = max(config.success_rate - 0.1, 0.0)
            config.last_error = result.error_message
    
    return result


@router.get("/presets",
    summary="Get Configuration Presets",
    description="Get template configurations for popular LLM providers",
    response_model=Dict[str, Any])
async def get_configuration_presets(
    category: Optional[str] = Query(None, description="Filter by category"),
    popular_only: bool = Query(False, description="Show only popular presets")
) -> Dict[str, Any]:
    """Get configuration presets for popular providers."""
    presets = list(PROVIDER_PRESETS.values())
    
    # Apply filters
    if category:
        presets = [p for p in presets if p.category == category]
    
    if popular_only:
        presets = [p for p in presets if p.popular]
    
    # Get available categories
    categories = list(set(p.category for p in PROVIDER_PRESETS.values()))
    
    return {
        "presets": [preset.dict() for preset in presets],
        "available_categories": categories,
        "total": len(presets)
    }
