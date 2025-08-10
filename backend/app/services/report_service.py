"""
Report Service - Generate and manage test reports in various formats.
"""

import asyncio
import json
import csv
import io
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging

from app.core.config import get_settings
from app.models.test_result import TestResult, TestSession, TestResultSummary
from app.models.failure_type import FailureReport, FailureClassification

logger = logging.getLogger(__name__)
settings = get_settings()


class ReportService:
    """Service for generating comprehensive test reports."""
    
    def __init__(self):
        self.reports_dir = Path(settings.REPORTS_DIRECTORY if hasattr(settings, 'REPORTS_DIRECTORY') else "./data/reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    async def generate_session_report(
        self,
        session_id: str,
        format: str = "json",
        include_details: bool = True,
        include_charts: bool = False
    ) -> Dict[str, Any]:
        """Generate a comprehensive report for a test session."""
        try:
            # TODO: Fetch session data from database
            session_data = await self._get_session_data(session_id)
            
            if not session_data:
                raise ValueError(f"Session {session_id} not found")
            
            # Generate report based on format
            if format.lower() == "json":
                return await self._generate_json_report(session_data, include_details)
            elif format.lower() == "csv":
                return await self._generate_csv_report(session_data)
            elif format.lower() == "html":
                return await self._generate_html_report(session_data, include_charts)
            elif format.lower() == "pdf":
                return await self._generate_pdf_report(session_data, include_charts)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Error generating report for session {session_id}: {e}")
            raise
    
    async def generate_batch_report(
        self,
        session_ids: List[str],
        format: str = "json",
        include_details: bool = True
    ) -> Dict[str, Any]:
        """Generate a report for multiple test sessions."""
        try:
            # Fetch data for all sessions
            sessions_data = []
            for session_id in session_ids:
                session_data = await self._get_session_data(session_id)
                if session_data:
                    sessions_data.append(session_data)
            
            if not sessions_data:
                raise ValueError("No valid sessions found")
            
            # Aggregate data
            aggregated_data = await self._aggregate_sessions_data(sessions_data)
            
            # Generate report
            if format.lower() == "json":
                return await self._generate_batch_json_report(aggregated_data, include_details)
            elif format.lower() == "csv":
                return await self._generate_batch_csv_report(aggregated_data)
            else:
                raise ValueError(f"Unsupported format for batch report: {format}")
                
        except Exception as e:
            logger.error(f"Error generating batch report: {e}")
            raise
    
    async def generate_trend_report(
        self,
        days: int = 7,
        format: str = "json"
    ) -> Dict[str, Any]:
        """Generate a trend analysis report."""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # TODO: Fetch trend data from database
            trend_data = await self._get_trend_data(start_date, end_date)
            
            if format.lower() == "json":
                return await self._generate_trend_json_report(trend_data, days)
            else:
                raise ValueError(f"Unsupported format for trend report: {format}")
                
        except Exception as e:
            logger.error(f"Error generating trend report: {e}")
            raise
    
    async def _get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Fetch session data from database."""
        # TODO: Implement database query
        # This is a placeholder implementation
        return {
            "session": {
                "id": session_id,
                "name": f"Test Session {session_id}",
                "status": "completed",
                "start_time": datetime.utcnow().isoformat(),
                "end_time": datetime.utcnow().isoformat(),
                "total_tests": 100,
                "passed_tests": 85,
                "failed_tests": 15,
                "model_name": "gpt-4o-mini",
                "provider": "openai"
            },
            "results": [
                {
                    "id": f"result_{i}",
                    "test_type": "prompt_injection",
                    "passed": i % 6 != 0,  # 85% pass rate
                    "execution_time": 1.5 + (i * 0.1),
                    "tokens_used": 150 + i,
                    "vulnerability_detected": i % 6 == 0,
                    "risk_level": "high" if i % 6 == 0 else "low"
                }
                for i in range(100)
            ],
            "failures": [
                {
                    "failure_type": "prompt_injection",
                    "count": 10,
                    "severity": "high"
                },
                {
                    "failure_type": "malformed_response",
                    "count": 5,
                    "severity": "medium"
                }
            ]
        }
    
    async def _get_trend_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Fetch trend data from database."""
        # TODO: Implement database query
        # This is a placeholder implementation
        days = (end_date - start_date).days
        return {
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": days
            },
            "daily_stats": [
                {
                    "date": (start_date + timedelta(days=i)).isoformat(),
                    "total_tests": 50 + i * 10,
                    "pass_rate": 0.85 + (i * 0.01),
                    "failure_rate": 0.15 - (i * 0.01),
                    "avg_response_time": 1.5 + (i * 0.1)
                }
                for i in range(days)
            ],
            "trends": {
                "test_volume_trend": "increasing",
                "quality_trend": "improving",
                "performance_trend": "stable"
            }
        }
    
    async def _aggregate_sessions_data(self, sessions_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate data from multiple sessions."""
        total_tests = sum(session["session"]["total_tests"] for session in sessions_data)
        total_passed = sum(session["session"]["passed_tests"] for session in sessions_data)
        total_failed = sum(session["session"]["failed_tests"] for session in sessions_data)
        
        # Aggregate failure types
        failure_aggregation = {}
        for session in sessions_data:
            for failure in session.get("failures", []):
                failure_type = failure["failure_type"]
                if failure_type not in failure_aggregation:
                    failure_aggregation[failure_type] = {"count": 0, "severity": failure["severity"]}
                failure_aggregation[failure_type]["count"] += failure["count"]
        
        return {
            "summary": {
                "total_sessions": len(sessions_data),
                "total_tests": total_tests,
                "total_passed": total_passed,
                "total_failed": total_failed,
                "overall_pass_rate": total_passed / total_tests if total_tests > 0 else 0
            },
            "sessions": [session["session"] for session in sessions_data],
            "aggregated_failures": failure_aggregation,
            "all_results": [result for session in sessions_data for result in session.get("results", [])]
        }
    
    async def _generate_json_report(self, session_data: Dict[str, Any], include_details: bool) -> Dict[str, Any]:
        """Generate JSON format report."""
        report = {
            "report_type": "session_report",
            "generated_at": datetime.utcnow().isoformat(),
            "session": session_data["session"],
            "summary": {
                "total_tests": session_data["session"]["total_tests"],
                "passed_tests": session_data["session"]["passed_tests"],
                "failed_tests": session_data["session"]["failed_tests"],
                "pass_rate": session_data["session"]["passed_tests"] / session_data["session"]["total_tests"],
                "failure_types": session_data.get("failures", [])
            }
        }
        
        if include_details:
            report["detailed_results"] = session_data.get("results", [])
            report["performance_metrics"] = {
                "avg_execution_time": sum(r["execution_time"] for r in session_data.get("results", [])) / len(session_data.get("results", [])) if session_data.get("results") else 0,
                "total_tokens_used": sum(r["tokens_used"] for r in session_data.get("results", [])),
                "vulnerabilities_detected": len([r for r in session_data.get("results", []) if r.get("vulnerability_detected")])
            }
        
        return report
    
    async def _generate_csv_report(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate CSV format report."""
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write headers
        writer.writerow([
            "Test ID", "Test Type", "Status", "Execution Time", 
            "Tokens Used", "Vulnerability Detected", "Risk Level"
        ])
        
        # Write data
        for result in session_data.get("results", []):
            writer.writerow([
                result["id"],
                result["test_type"],
                "PASS" if result["passed"] else "FAIL",
                result["execution_time"],
                result["tokens_used"],
                result.get("vulnerability_detected", False),
                result.get("risk_level", "unknown")
            ])
        
        csv_content = output.getvalue()
        output.close()
        
        return {
            "format": "csv",
            "content": csv_content,
            "filename": f"session_{session_data['session']['id']}_report.csv"
        }
    
    async def _generate_html_report(self, session_data: Dict[str, Any], include_charts: bool) -> Dict[str, Any]:
        """Generate HTML format report."""
        # TODO: Implement HTML report generation with templates
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Report - Session {session_data['session']['id']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background: white; border-radius: 3px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Test Report - Session {session_data['session']['id']}</h1>
            <div class="summary">
                <h2>Summary</h2>
                <div class="metric">Total Tests: {session_data['session']['total_tests']}</div>
                <div class="metric">Passed: {session_data['session']['passed_tests']}</div>
                <div class="metric">Failed: {session_data['session']['failed_tests']}</div>
                <div class="metric">Pass Rate: {session_data['session']['passed_tests'] / session_data['session']['total_tests']:.2%}</div>
            </div>
            <h2>Test Results</h2>
            <table>
                <tr><th>Test ID</th><th>Type</th><th>Status</th><th>Execution Time</th><th>Tokens</th></tr>
        """
        
        for result in session_data.get("results", [])[:20]:  # Limit for demo
            status = "PASS" if result["passed"] else "FAIL"
            html_content += f"""
                <tr>
                    <td>{result['id']}</td>
                    <td>{result['test_type']}</td>
                    <td>{status}</td>
                    <td>{result['execution_time']:.2f}s</td>
                    <td>{result['tokens_used']}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        return {
            "format": "html",
            "content": html_content,
            "filename": f"session_{session_data['session']['id']}_report.html"
        }
    
    async def _generate_pdf_report(self, session_data: Dict[str, Any], include_charts: bool) -> Dict[str, Any]:
        """Generate PDF format report."""
        # TODO: Implement PDF generation (requires reportlab or similar)
        # For now, return a placeholder
        return {
            "format": "pdf",
            "content": "PDF generation not implemented yet",
            "filename": f"session_{session_data['session']['id']}_report.pdf",
            "error": "PDF generation requires additional dependencies"
        }
    
    async def _generate_batch_json_report(self, aggregated_data: Dict[str, Any], include_details: bool) -> Dict[str, Any]:
        """Generate JSON batch report."""
        report = {
            "report_type": "batch_report",
            "generated_at": datetime.utcnow().isoformat(),
            "summary": aggregated_data["summary"],
            "sessions": aggregated_data["sessions"],
            "aggregated_failures": aggregated_data["aggregated_failures"]
        }
        
        if include_details:
            report["all_results"] = aggregated_data["all_results"]
        
        return report
    
    async def _generate_batch_csv_report(self, aggregated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate CSV batch report."""
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write summary
        writer.writerow(["Batch Report Summary"])
        writer.writerow(["Total Sessions", aggregated_data["summary"]["total_sessions"]])
        writer.writerow(["Total Tests", aggregated_data["summary"]["total_tests"]])
        writer.writerow(["Overall Pass Rate", f"{aggregated_data['summary']['overall_pass_rate']:.2%}"])
        writer.writerow([])  # Empty row
        
        # Write session details
        writer.writerow(["Session ID", "Name", "Total Tests", "Passed", "Failed", "Pass Rate"])
        for session in aggregated_data["sessions"]:
            pass_rate = session["passed_tests"] / session["total_tests"] if session["total_tests"] > 0 else 0
            writer.writerow([
                session["id"],
                session["name"],
                session["total_tests"],
                session["passed_tests"],
                session["failed_tests"],
                f"{pass_rate:.2%}"
            ])
        
        csv_content = output.getvalue()
        output.close()
        
        return {
            "format": "csv",
            "content": csv_content,
            "filename": "batch_report.csv"
        }
    
    async def _generate_trend_json_report(self, trend_data: Dict[str, Any], days: int) -> Dict[str, Any]:
        """Generate JSON trend report."""
        return {
            "report_type": "trend_report",
            "generated_at": datetime.utcnow().isoformat(),
            "period_days": days,
            "period": trend_data["period"],
            "daily_statistics": trend_data["daily_stats"],
            "trends": trend_data["trends"],
            "insights": await self._generate_trend_insights(trend_data)
        }
    
    async def _generate_trend_insights(self, trend_data: Dict[str, Any]) -> List[str]:
        """Generate insights from trend data."""
        insights = []
        
        daily_stats = trend_data["daily_stats"]
        if len(daily_stats) >= 2:
            first_day = daily_stats[0]
            last_day = daily_stats[-1]
            
            # Test volume trend
            volume_change = (last_day["total_tests"] - first_day["total_tests"]) / first_day["total_tests"]
            if volume_change > 0.1:
                insights.append(f"Test volume increased by {volume_change:.1%} over the period")
            elif volume_change < -0.1:
                insights.append(f"Test volume decreased by {abs(volume_change):.1%} over the period")
            
            # Quality trend
            quality_change = last_day["pass_rate"] - first_day["pass_rate"]
            if quality_change > 0.05:
                insights.append(f"Test pass rate improved by {quality_change:.1%}")
            elif quality_change < -0.05:
                insights.append(f"Test pass rate declined by {abs(quality_change):.1%}")
        
        return insights
