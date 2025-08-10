"""
Input Validators - Validation utilities for API inputs and data models.
"""

import re
import ipaddress
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, date
from email_validator import validate_email, EmailNotValidError
from urllib.parse import urlparse
from pydantic import ValidationError
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom validation error."""
    def __init__(self, message: str, field: Optional[str] = None):
        self.message = message
        self.field = field
        super().__init__(message)


class InputValidators:
    """Collection of input validation utilities."""
    
    @staticmethod
    def validate_string_length(
        value: str, 
        min_length: int = 0, 
        max_length: int = None,
        field_name: str = "field"
    ) -> str:
        """Validate string length constraints."""
        if not isinstance(value, str):
            raise ValidationError(f"{field_name} must be a string", field_name)
        
        if len(value) < min_length:
            raise ValidationError(
                f"{field_name} must be at least {min_length} characters long", 
                field_name
            )
        
        if max_length and len(value) > max_length:
            raise ValidationError(
                f"{field_name} must be no more than {max_length} characters long", 
                field_name
            )
        
        return value.strip()
    
    @staticmethod
    def validate_email_address(email: str, field_name: str = "email") -> str:
        """Validate email address format."""
        try:
            # Use email-validator library for comprehensive validation
            validation_result = validate_email(email)
            return validation_result.email
        except EmailNotValidError as e:
            raise ValidationError(f"Invalid {field_name}: {str(e)}", field_name)
    
    @staticmethod
    def validate_url(url: str, field_name: str = "url", allowed_schemes: List[str] = None) -> str:
        """Validate URL format and scheme."""
        if not url:
            raise ValidationError(f"{field_name} cannot be empty", field_name)
        
        try:
            parsed = urlparse(url)
            
            if not parsed.scheme:
                raise ValidationError(f"{field_name} must include a scheme (http/https)", field_name)
            
            if not parsed.netloc:
                raise ValidationError(f"{field_name} must include a valid domain", field_name)
            
            if allowed_schemes and parsed.scheme.lower() not in allowed_schemes:
                raise ValidationError(
                    f"{field_name} scheme must be one of: {', '.join(allowed_schemes)}", 
                    field_name
                )
            
            return url
            
        except Exception as e:
            raise ValidationError(f"Invalid {field_name} format: {str(e)}", field_name)
    
    @staticmethod
    def validate_ip_address(ip: str, field_name: str = "ip_address") -> str:
        """Validate IP address (IPv4 or IPv6)."""
        try:
            ipaddress.ip_address(ip)
            return ip
        except ipaddress.AddressValueError:
            raise ValidationError(f"Invalid {field_name} format", field_name)
    
    @staticmethod
    def validate_numeric_range(
        value: Union[int, float], 
        min_value: Union[int, float] = None,
        max_value: Union[int, float] = None,
        field_name: str = "value"
    ) -> Union[int, float]:
        """Validate numeric value within specified range."""
        if not isinstance(value, (int, float)):
            raise ValidationError(f"{field_name} must be a number", field_name)
        
        if min_value is not None and value < min_value:
            raise ValidationError(f"{field_name} must be at least {min_value}", field_name)
        
        if max_value is not None and value > max_value:
            raise ValidationError(f"{field_name} must be no more than {max_value}", field_name)
        
        return value
    
    @staticmethod
    def validate_date_format(
        date_str: str, 
        format: str = "%Y-%m-%d",
        field_name: str = "date"
    ) -> datetime:
        """Validate date string format."""
        try:
            return datetime.strptime(date_str, format)
        except ValueError:
            raise ValidationError(
                f"Invalid {field_name} format. Expected format: {format}", 
                field_name
            )
    
    @staticmethod
    def validate_date_range(
        date_value: Union[str, datetime, date],
        min_date: Union[str, datetime, date] = None,
        max_date: Union[str, datetime, date] = None,
        field_name: str = "date"
    ) -> datetime:
        """Validate date within specified range."""
        # Convert to datetime if string
        if isinstance(date_value, str):
            date_value = InputValidators.validate_date_format(date_value, field_name=field_name)
        elif isinstance(date_value, date) and not isinstance(date_value, datetime):
            date_value = datetime.combine(date_value, datetime.min.time())
        
        # Convert comparison dates
        if isinstance(min_date, str):
            min_date = datetime.strptime(min_date, "%Y-%m-%d")
        elif isinstance(min_date, date) and not isinstance(min_date, datetime):
            min_date = datetime.combine(min_date, datetime.min.time())
        
        if isinstance(max_date, str):
            max_date = datetime.strptime(max_date, "%Y-%m-%d")
        elif isinstance(max_date, date) and not isinstance(max_date, datetime):
            max_date = datetime.combine(max_date, datetime.max.time())
        
        # Validate range
        if min_date and date_value < min_date:
            raise ValidationError(f"{field_name} cannot be before {min_date.date()}", field_name)
        
        if max_date and date_value > max_date:
            raise ValidationError(f"{field_name} cannot be after {max_date.date()}", field_name)
        
        return date_value
    
    @staticmethod
    def validate_password_strength(
        password: str, 
        min_length: int = 8,
        require_uppercase: bool = True,
        require_lowercase: bool = True,
        require_digits: bool = True,
        require_special: bool = True,
        field_name: str = "password"
    ) -> str:
        """Validate password strength requirements."""
        if len(password) < min_length:
            raise ValidationError(
                f"{field_name} must be at least {min_length} characters long", 
                field_name
            )
        
        requirements = []
        
        if require_uppercase and not re.search(r'[A-Z]', password):
            requirements.append("at least one uppercase letter")
        
        if require_lowercase and not re.search(r'[a-z]', password):
            requirements.append("at least one lowercase letter")
        
        if require_digits and not re.search(r'\d', password):
            requirements.append("at least one digit")
        
        if require_special and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            requirements.append("at least one special character")
        
        if requirements:
            raise ValidationError(
                f"{field_name} must contain {', '.join(requirements)}", 
                field_name
            )
        
        return password
    
    @staticmethod
    def validate_json_structure(
        json_data: Any, 
        required_fields: List[str] = None,
        field_name: str = "json_data"
    ) -> Dict[str, Any]:
        """Validate JSON structure and required fields."""
        if not isinstance(json_data, dict):
            raise ValidationError(f"{field_name} must be a JSON object", field_name)
        
        if required_fields:
            missing_fields = []
            for field in required_fields:
                if field not in json_data:
                    missing_fields.append(field)
            
            if missing_fields:
                raise ValidationError(
                    f"{field_name} missing required fields: {', '.join(missing_fields)}", 
                    field_name
                )
        
        return json_data
    
    @staticmethod
    def validate_file_extension(
        filename: str, 
        allowed_extensions: List[str],
        field_name: str = "filename"
    ) -> str:
        """Validate file extension."""
        if not filename:
            raise ValidationError(f"{field_name} cannot be empty", field_name)
        
        # Extract extension
        extension = filename.lower().split('.')[-1] if '.' in filename else ''
        
        # Normalize allowed extensions (remove dots if present)
        normalized_allowed = [ext.lstrip('.').lower() for ext in allowed_extensions]
        
        if extension not in normalized_allowed:
            raise ValidationError(
                f"{field_name} must have one of these extensions: {', '.join(allowed_extensions)}", 
                field_name
            )
        
        return filename
    
    @staticmethod
    def validate_file_size(
        file_size: int, 
        max_size: int,
        field_name: str = "file"
    ) -> int:
        """Validate file size."""
        if file_size > max_size:
            max_size_mb = max_size / (1024 * 1024)
            raise ValidationError(
                f"{field_name} size exceeds maximum allowed size of {max_size_mb:.1f}MB", 
                field_name
            )
        
        return file_size
    
    @staticmethod
    def validate_uuid(uuid_str: str, field_name: str = "id") -> str:
        """Validate UUID format."""
        import uuid
        try:
            uuid.UUID(uuid_str)
            return uuid_str
        except ValueError:
            raise ValidationError(f"Invalid {field_name} format (must be UUID)", field_name)
    
    @staticmethod
    def validate_choice(
        value: Any, 
        choices: List[Any],
        field_name: str = "value"
    ) -> Any:
        """Validate value is one of allowed choices."""
        if value not in choices:
            raise ValidationError(
                f"{field_name} must be one of: {', '.join(map(str, choices))}", 
                field_name
            )
        
        return value
    
    @staticmethod
    def validate_regex_pattern(
        value: str, 
        pattern: str,
        error_message: str = None,
        field_name: str = "value"
    ) -> str:
        """Validate string matches regex pattern."""
        if not re.match(pattern, value):
            if error_message:
                raise ValidationError(error_message, field_name)
            else:
                raise ValidationError(f"{field_name} format is invalid", field_name)
        
        return value
    
    @staticmethod
    def sanitize_html_input(html_input: str, field_name: str = "html") -> str:
        """Sanitize HTML input to prevent XSS."""
        # Simple HTML sanitization - for production use bleach library
        import html
        
        # Escape HTML entities
        sanitized = html.escape(html_input)
        
        # Remove common problematic patterns
        dangerous_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'<iframe[^>]*>.*?</iframe>',
            r'<object[^>]*>.*?</object>',
            r'<embed[^>]*>.*?</embed>'
        ]
        
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        return sanitized
    
    @staticmethod
    def validate_api_key_format(api_key: str, field_name: str = "api_key") -> str:
        """Validate API key format."""
        if not api_key:
            raise ValidationError(f"{field_name} cannot be empty", field_name)
        
        # Basic API key format validation
        if len(api_key) < 10:
            raise ValidationError(f"{field_name} is too short", field_name)
        
        if len(api_key) > 200:
            raise ValidationError(f"{field_name} is too long", field_name)
        
        # Check for obvious invalid patterns
        if api_key.isspace():
            raise ValidationError(f"{field_name} cannot be only whitespace", field_name)
        
        return api_key.strip()
    
    @staticmethod
    def validate_batch_size(
        batch_size: int, 
        max_batch_size: int = 1000,
        field_name: str = "batch_size"
    ) -> int:
        """Validate batch processing size."""
        if batch_size <= 0:
            raise ValidationError(f"{field_name} must be greater than 0", field_name)
        
        if batch_size > max_batch_size:
            raise ValidationError(
                f"{field_name} cannot exceed {max_batch_size}", 
                field_name
            )
        
        return batch_size
    
    @staticmethod
    def validate_test_case_data(test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Validate test case data structure."""
        required_fields = ["name", "prompt", "category"]
        
        # Check required fields
        for field in required_fields:
            if field not in test_case:
                raise ValidationError(f"Test case missing required field: {field}")
            
            if not test_case[field] or str(test_case[field]).strip() == "":
                raise ValidationError(f"Test case field '{field}' cannot be empty")
        
        # Validate specific fields
        InputValidators.validate_string_length(
            test_case["name"], 
            min_length=1, 
            max_length=200, 
            field_name="name"
        )
        
        InputValidators.validate_string_length(
            test_case["prompt"], 
            min_length=1, 
            max_length=10000, 
            field_name="prompt"
        )
        
        # Validate optional numeric fields
        if "max_tokens" in test_case:
            InputValidators.validate_numeric_range(
                test_case["max_tokens"], 
                min_value=1, 
                max_value=100000,
                field_name="max_tokens"
            )
        
        if "temperature" in test_case:
            InputValidators.validate_numeric_range(
                test_case["temperature"], 
                min_value=0.0, 
                max_value=2.0,
                field_name="temperature"
            )
        
        return test_case


class ValidationUtils:
    """Additional validation utilities."""
    
    @staticmethod
    def create_validation_summary(errors: List[ValidationError]) -> Dict[str, Any]:
        """Create a summary of validation errors."""
        return {
            "is_valid": len(errors) == 0,
            "error_count": len(errors),
            "errors": [
                {
                    "field": error.field,
                    "message": error.message
                }
                for error in errors
            ]
        }
    
    @staticmethod
    def validate_multiple_fields(
        data: Dict[str, Any], 
        validations: Dict[str, callable]
    ) -> Dict[str, Any]:
        """Validate multiple fields with different validators."""
        errors = []
        validated_data = {}
        
        for field, validator in validations.items():
            try:
                if field in data:
                    validated_data[field] = validator(data[field])
                else:
                    validated_data[field] = None
            except ValidationError as e:
                errors.append(e)
        
        return {
            "validated_data": validated_data,
            "validation_summary": ValidationUtils.create_validation_summary(errors)
        }
