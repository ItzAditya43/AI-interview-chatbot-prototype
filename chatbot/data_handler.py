"""
Manages candidate data and interview session information.
"""

import json
import os
import glob
import csv # Import the csv module
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from utils.security import generate_session_id # Assuming generate_session_id is now in utils.security
from config.settings import MAX_TECHNICAL_QUESTIONS # Import MAX_TECHNICAL_QUESTIONS

class DataHandler:
    """Handles saving, loading, and managing interview session data."""

    def __init__(self, base_data_dir: str = "data/sessions"):
        self.base_data_dir = base_data_dir
        os.makedirs(self.base_data_dir, exist_ok=True)

    def _get_session_filepath(self, session_id: str) -> str:
        """Generate filepath for a session data file."""
        today_str = datetime.now().strftime('%Y-%m-%d')
        session_dir = os.path.join(self.base_data_dir, today_str)
        os.makedirs(session_dir, exist_ok=True)
        return os.path.join(session_dir, f'{session_id}.json')

    def save_session_data(self, session_id: str, data: Dict[str, Any]):
        """Save interview session data to a JSON file."""
        filepath = self._get_session_filepath(session_id)
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            # print(f"Session data saved: {filepath}") # Optional logging
        except Exception as e:
            print(f"Error saving session data to {filepath}: {e}") # Basic error reporting

    def load_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load interview session data from a JSON file."""
        # This simplified load assumes you know the date or search across dates.
        # A more robust version would search across recent date directories.
        today_str = datetime.now().strftime('%Y-%m-%d')
        filepath = os.path.join(self.base_data_dir, today_str, f'{session_id}.json')
        
        if not os.path.exists(filepath):
             # If not found in today's directory, search recent directories (e.g., last 7 days)
             for i in range(7):
                  past_date = datetime.now() - timedelta(days=i)
                  past_date_str = past_date.strftime('%Y-%m-%d')
                  past_filepath = os.path.join(self.base_data_dir, past_date_str, f'{session_id}.json')
                  if os.path.exists(past_filepath):
                       filepath = past_filepath
                       break
             else:
                  return None # Session not found

        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading session data from {filepath}: {e}")
            return None

    def export_candidate_profile(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Export candidate profile data for a session."""
        session_data = self.load_session_data(session_id)
        if session_data and "user_data" in session_data:
            # Return just the relevant candidate info fields from user_data
            user_data = session_data["user_data"]
            candidate_info = {
                "full_name": user_data.get("full_name", ""),
                "email": user_data.get("email", ""),
                "phone": user_data.get("phone", ""),
                "years_experience": user_data.get("years_experience", 0),
                "desired_positions": user_data.get("desired_positions", []),
                "current_location": user_data.get("current_location", ""),
                "tech_stack": user_data.get("tech_stack", [])
            }
            return candidate_info
        return None

    def find_sessions_by_email(self, email: str) -> List[str]:
        """Find all session IDs associated with a given candidate email address."""
        matching_session_ids = []
        # Walk through date directories to find session files
        for date_dir in glob.glob(os.path.join(self.base_data_dir, '*')):
             if os.path.isdir(date_dir):
                  for session_file in glob.glob(os.path.join(date_dir, '*.json')):
                       try:
                           with open(session_file, 'r') as f:
                                session_data = json.load(f)
                                # Check if user_data exists and contains the target email
                                if "user_data" in session_data and session_data["user_data"].get("email", "").lower() == email.lower():
                                     session_id = os.path.splitext(os.path.basename(session_file))[0]
                                     matching_session_ids.append(session_id)
                       except Exception as e:
                            print(f"Error reading session file {session_file} during email search: {e}")
                            continue
        # Return unique session IDs (though they should be unique by filename)
        return list(set(matching_session_ids))

    def list_all_sessions(self) -> List[str]:
        """List all available session IDs."""
        session_ids = []
        # Walk through date directories to find session files
        for date_dir in glob.glob(os.path.join(self.base_data_dir, '*')):
             if os.path.isdir(date_dir):
                  for session_file in glob.glob(os.path.join(date_dir, '*.json')):
                       session_id = os.path.splitext(os.path.basename(session_file))[0]
                       session_ids.append(session_id)
        return session_ids

    def cleanup_old_sessions(self, days_old: int = 30):
        """Remove session data files older than a specified number of days."""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        for date_dir in glob.glob(os.path.join(self.base_data_dir, '*')):
             if os.path.isdir(date_dir):
                  try:
                       # Attempt to parse the directory name as a date
                       dir_date = datetime.strptime(os.path.basename(date_dir), '%Y-%m-%d')
                       if dir_date < cutoff_date:
                            print(f"Cleaning up old session directory: {date_dir}")
                            # Remove the directory and its contents
                            import shutil
                            shutil.rmtree(date_dir)
                  except ValueError:
                       # Ignore directories that don't match the date format
                       pass
                  except Exception as e:
                       print(f"Error during cleanup of {date_dir}: {e}")

    def export_sessions_to_csv(self) -> str:
        """
        Exports all saved interview sessions to a CSV string.

        Returns:
            str: A string containing the data in CSV format.
        """
        all_session_ids = self.list_all_sessions()
        csv_rows = []

        # Define header row - Candidate Info + columns for each technical question and its follow-up
        # We'll create columns for up to MAX_TECHNICAL_QUESTIONS
        header = ["Session ID", "Full Name", "Email", "Phone", "Years Experience", "Tech Stack"]
        for i in range(MAX_TECHNICAL_QUESTIONS):
            header.append(f"Question {i+1}")
            header.append(f"Answer {i+1}")
            header.append(f"Feedback {i+1}")
            header.append(f"Follow-up {i+1}")
            header.append(f"Follow-up Answer {i+1}")
            header.append(f"Follow-up Feedback {i+1}")

        csv_rows.append(header)

        for session_id in all_session_ids:
            session_data = self.load_session_data(session_id)
            if session_data and "user_data" in session_data:
                user_data = session_data["user_data"]

                # Start building the row with candidate info
                row = [
                    session_id,
                    user_data.get("full_name", ""),
                    user_data.get("email", ""),
                    user_data.get("phone", ""),
                    user_data.get("years_experience", ""), # Include as string for CSV
                    ", ".join(user_data.get("tech_stack", []))
                ]

                # Add data for technical questions
                technical_questions = user_data.get("technical_questions", [])
                for i in range(MAX_TECHNICAL_QUESTIONS):
                    if i < len(technical_questions):
                         question = technical_questions[i]
                         # Handle both dictionary and object format for questions if necessary,
                         # but assuming they are loaded as TechnicalQuestion objects now
                         row.append(question.question if hasattr(question, 'question') else question.get('question', ''))
                         row.append(question.answer if hasattr(question, 'answer') else question.get('answer', ''))
                         row.append(question.feedback if hasattr(question, 'feedback') else question.get('feedback', ''))
                         row.append(question.generated_follow_up_question if hasattr(question, 'generated_follow_up_question') else question.get('generated_follow_up_question', ''))
                         row.append(question.follow_up_answer if hasattr(question, 'follow_up_answer') else question.get('follow_up_answer', ''))
                         row.append(question.follow_up_feedback if hasattr(question, 'follow_up_feedback') else question.get('follow_up_feedback', ''))
                    else:
                         # Pad with empty strings if fewer than MAX_TECHNICAL_QUESTIONS
                         row.extend(["", "", "", "", "", ""])

                csv_rows.append(row)

        # Write rows to a string in CSV format
        import io # Import io module
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerows(csv_rows)

        return output.getvalue()

    def save_candidate_data(self, candidate_id: str, data: Dict[str, Any]):
        """Save candidate interview data."""
        # TODO: Implement candidate data saving
        pass

    def load_candidate_data(self, candidate_id: str) -> Dict[str, Any]:
        """Load candidate interview data."""
        # TODO: Implement candidate data loading
        return {}

    def generate_analytics(self) -> Dict[str, Any]:
        """
        Generates comprehensive analytics from all saved interview sessions.

        Returns:
            Dict[str, Any]: A dictionary containing detailed analytics data.
        """
        all_session_ids = self.list_all_sessions()
        total_sessions = len(all_session_ids)
        completed_sessions = 0
        
        # Initialize analytics counters and aggregators
        analytics = {
            "session_metrics": {
                "total_sessions": total_sessions,
                "completed_sessions": 0,
                "completion_rate": "0%",
                "average_session_duration": 0,
                "sessions_by_date": {},
            },
            "tech_stack_metrics": {
                "distribution": {},
                "popular_combinations": {},
                "average_skills_per_candidate": 0,
            },
            "question_metrics": {
                "total_questions_asked": 0,
                "questions_by_difficulty": {"easy": 0, "medium": 0, "hard": 0},
                "questions_by_topic": {},
                "average_questions_per_session": 0,
                "follow_up_rate": 0,
            },
            "performance_metrics": {
                "average_answer_length": 0,
                "feedback_distribution": {
                    "positive": 0,
                    "neutral": 0,
                    "needs_improvement": 0
                },
                "topic_performance": {},
                "difficulty_performance": {
                    "easy": {"correct": 0, "total": 0},
                    "medium": {"correct": 0, "total": 0},
                    "hard": {"correct": 0, "total": 0}
                }
            },
            "time_metrics": {
                "average_time_per_question": 0,
                "average_time_per_session": 0,
                "time_distribution_by_topic": {}
            }
        }
        
        total_session_duration = 0
        total_questions = 0
        total_follow_ups = 0
        total_answer_length = 0
        total_skills = 0
        
        for session_id in all_session_ids:
            session_data = self.load_session_data(session_id)
            if not session_data or "user_data" not in session_data:
                continue
            
            user_data = session_data["user_data"]
            
            # Track session completion
            if user_data.get("conversation_complete", False):
                completed_sessions += 1
            
            # Track session date
            session_date = datetime.fromisoformat(user_data.get("timestamp", "")).strftime("%Y-%m-%d")
            analytics["session_metrics"]["sessions_by_date"][session_date] = \
                analytics["session_metrics"]["sessions_by_date"].get(session_date, 0) + 1
            
            # Track tech stack metrics
            tech_stack = user_data.get("tech_stack", [])
            total_skills += len(tech_stack)
            
            # Update tech stack distribution
            for tech in tech_stack:
                analytics["tech_stack_metrics"]["distribution"][tech] = \
                    analytics["tech_stack_metrics"]["distribution"].get(tech, 0) + 1
                
            # Track popular tech stack combinations (pairs)
            for i in range(len(tech_stack)):
                for j in range(i + 1, len(tech_stack)):
                    combo = tuple(sorted([tech_stack[i], tech_stack[j]]))
                    analytics["tech_stack_metrics"]["popular_combinations"][combo] = \
                        analytics["tech_stack_metrics"]["popular_combinations"].get(combo, 0) + 1
                    
            # Process technical questions
            questions = user_data.get("technical_questions", [])
            total_questions += len(questions)
            
            for question in questions:
                # Track question difficulty
                difficulty = question.get("difficulty", "medium").lower()
                analytics["question_metrics"]["questions_by_difficulty"][difficulty] = \
                    analytics["question_metrics"]["questions_by_difficulty"].get(difficulty, 0) + 1
                
                # Track question topics
                topic = question.get("topic", "unknown")
                analytics["question_metrics"]["questions_by_topic"][topic] = \
                    analytics["question_metrics"]["questions_by_topic"].get(topic, 0) + 1
                
                # Track follow-up questions
                if question.get("generated_follow_up_question"):
                    total_follow_ups += 1
                
                # Track answer length
                if question.get("answer"):
                    total_answer_length += len(question["answer"])
                
                # Track feedback sentiment (simple heuristic)
                feedback = question.get("feedback", "").lower()
                if feedback:
                    if any(word in feedback for word in ["excellent", "great", "good", "well done"]):
                        analytics["performance_metrics"]["feedback_distribution"]["positive"] += 1
                    elif any(word in feedback for word in ["needs", "improve", "better", "could be"]):
                        analytics["performance_metrics"]["feedback_distribution"]["needs_improvement"] += 1
                    else:
                        analytics["performance_metrics"]["feedback_distribution"]["neutral"] += 1
                    
                # Track topic performance
                if topic not in analytics["performance_metrics"]["topic_performance"]:
                    analytics["performance_metrics"]["topic_performance"][topic] = {
                        "correct": 0,
                        "total": 0
                    }
                analytics["performance_metrics"]["topic_performance"][topic]["total"] += 1
                
                # Simple heuristic for correct answers based on feedback
                if feedback and any(word in feedback for word in ["excellent", "great", "good", "correct"]):
                    analytics["performance_metrics"]["topic_performance"][topic]["correct"] += 1
                    analytics["performance_metrics"]["difficulty_performance"][difficulty]["correct"] += 1
                analytics["performance_metrics"]["difficulty_performance"][difficulty]["total"] += 1
                
        # Calculate averages and rates
        if total_sessions > 0:
            analytics["session_metrics"]["completion_rate"] = f"{(completed_sessions / total_sessions * 100):.2f}%"
            analytics["tech_stack_metrics"]["average_skills_per_candidate"] = total_skills / total_sessions
            analytics["question_metrics"]["average_questions_per_session"] = total_questions / total_sessions
            analytics["question_metrics"]["follow_up_rate"] = (total_follow_ups / total_questions * 100) if total_questions > 0 else 0
            analytics["performance_metrics"]["average_answer_length"] = total_answer_length / total_questions if total_questions > 0 else 0
            
        # Calculate time-based analytics
        total_session_duration_seconds = 0
        total_question_duration_seconds = 0
        question_count_with_time = 0

        for session_id in all_session_ids:
            session_data = self.load_session_data(session_id)
            if not session_data or "user_data" not in session_data:
                continue

            user_data = session_data["user_data"]

            # Calculate total session duration
            session_start = user_data.get("session_start_time")
            session_end = user_data.get("session_end_time")

            if session_start and session_end:
                try:
                    start_dt = datetime.fromisoformat(session_start)
                    end_dt = datetime.fromisoformat(session_end)
                    duration = (end_dt - start_dt).total_seconds()
                    total_session_duration_seconds += duration
                except (ValueError, TypeError) as e:
                    print(f"Error calculating session duration for {session_id}: {e}")

            # Calculate total question duration
            technical_questions = user_data.get("technical_questions", [])
            for question in technical_questions:
                question_start = question.get("start_time")
                question_end = question.get("end_time")

                if question_start and question_end:
                    try:
                        start_dt = datetime.fromisoformat(question_start)
                        end_dt = datetime.fromisoformat(question_end)
                        duration = (end_dt - start_dt).total_seconds()
                        total_question_duration_seconds += duration
                        question_count_with_time += 1
                    except (ValueError, TypeError) as e:
                        print(f"Error calculating question duration for session {session_id}, question: {question.get('question', 'N/A')}: {e}")

        # Store time-based analytics
        analytics["time_metrics"]["average_session_duration"] = (total_session_duration_seconds / total_sessions) if total_sessions > 0 else 0
        analytics["time_metrics"]["average_time_per_question"] = (total_question_duration_seconds / question_count_with_time) if question_count_with_time > 0 else 0

        # You could potentially add time_distribution_by_topic here by summing up question durations per topic
        # This would require iterating through questions again or modifying the loop above.
        # For now, I'll leave time_distribution_by_topic as 0 or calculate it separately if needed.

        # Convert average session duration to a human-readable format (e.g., HH:MM:SS)
        avg_sess_duration_sec = analytics["time_metrics"]["average_session_duration"]
        if avg_sess_duration_sec > 0:
            hours, remain = divmod(avg_sess_duration_sec, 3600)
            minutes, seconds = divmod(remain, 60)
            analytics["session_metrics"]["average_session_duration"] = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
        else:
            analytics["session_metrics"]["average_session_duration"] = "N/A"

        # Format average time per question
        avg_time_per_q_sec = analytics["time_metrics"]["average_time_per_question"]
        if avg_time_per_q_sec > 0:
            analytics["question_metrics"]["average_time_per_question"] = f"{avg_time_per_q_sec:.2f} seconds"
        else:
            analytics["question_metrics"]["average_time_per_question"] = "N/A"

        # Sort and limit popular combinations to top 10
        analytics["tech_stack_metrics"]["popular_combinations"] = dict(
            sorted(analytics["tech_stack_metrics"]["popular_combinations"].items(),
                   key=lambda x: x[1],
                   reverse=True)[:10]
        )
        
        # Calculate success rates for topics and difficulties
        for topic in analytics["performance_metrics"]["topic_performance"]:
            metrics = analytics["performance_metrics"]["topic_performance"][topic]
            if metrics["total"] > 0:
                metrics["success_rate"] = f"{(metrics['correct'] / metrics['total'] * 100):.2f}%"
                
        for difficulty in analytics["performance_metrics"]["difficulty_performance"]:
            metrics = analytics["performance_metrics"]["difficulty_performance"][difficulty]
            if metrics["total"] > 0:
                metrics["success_rate"] = f"{(metrics['correct'] / metrics['total'] * 100):.2f}%"
                
        return analytics 

    def export_analytics_to_csv(self, analytics_data: Dict[str, Any]) -> str:
        """
        Exports analytics data to CSV format.
        
        Args:
            analytics_data: The analytics data dictionary from generate_analytics()
            
        Returns:
            str: CSV formatted string containing the analytics data
        """
        import io
        import csv
        from datetime import datetime
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write timestamp
        writer.writerow(["Analytics Report", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        writer.writerow([])
        
        # Session Metrics
        writer.writerow(["Session Metrics"])
        writer.writerow(["Metric", "Value"])
        session_metrics = analytics_data["session_metrics"]
        writer.writerow(["Total Sessions", session_metrics["total_sessions"]])
        writer.writerow(["Completed Sessions", session_metrics["completed_sessions"]])
        writer.writerow(["Completion Rate", session_metrics["completion_rate"]])
        writer.writerow(["Average Session Duration", session_metrics["average_session_duration"]])
        writer.writerow([])
        
        # Sessions by Date
        writer.writerow(["Sessions by Date"])
        writer.writerow(["Date", "Count"])
        for date, count in sorted(session_metrics["sessions_by_date"].items()):
            writer.writerow([date, count])
        writer.writerow([])
        
        # Tech Stack Metrics
        writer.writerow(["Tech Stack Metrics"])
        writer.writerow(["Metric", "Value"])
        tech_metrics = analytics_data["tech_stack_metrics"]
        writer.writerow(["Average Skills per Candidate", tech_metrics["average_skills_per_candidate"]])
        writer.writerow([])
        
        # Tech Stack Distribution
        writer.writerow(["Tech Stack Distribution"])
        writer.writerow(["Technology", "Count"])
        for tech, count in sorted(tech_metrics["distribution"].items(), key=lambda x: x[1], reverse=True):
            writer.writerow([tech, count])
        writer.writerow([])
        
        # Popular Tech Stack Combinations
        writer.writerow(["Popular Tech Stack Combinations"])
        writer.writerow(["Combination", "Count"])
        for combo, count in tech_metrics["popular_combinations"].items():
            writer.writerow([", ".join(combo), count])
        writer.writerow([])
        
        # Question Metrics
        writer.writerow(["Question Metrics"])
        writer.writerow(["Metric", "Value"])
        question_metrics = analytics_data["question_metrics"]
        writer.writerow(["Total Questions Asked", question_metrics["total_questions_asked"]])
        writer.writerow(["Average Questions per Session", question_metrics["average_questions_per_session"]])
        writer.writerow(["Follow-up Rate", f"{question_metrics['follow_up_rate']:.2f}%"])
        writer.writerow(["Average Time per Question", question_metrics.get("average_time_per_question", "N/A")])
        writer.writerow([])
        
        # Questions by Difficulty
        writer.writerow(["Questions by Difficulty"])
        writer.writerow(["Difficulty", "Count"])
        for diff, count in question_metrics["questions_by_difficulty"].items():
            writer.writerow([diff, count])
        writer.writerow([])
        
        # Questions by Topic
        writer.writerow(["Questions by Topic"])
        writer.writerow(["Topic", "Count"])
        for topic, count in sorted(question_metrics["questions_by_topic"].items(), key=lambda x: x[1], reverse=True):
            writer.writerow([topic, count])
        writer.writerow([])
        
        # Performance Metrics
        writer.writerow(["Performance Metrics"])
        writer.writerow(["Metric", "Value"])
        perf_metrics = analytics_data["performance_metrics"]
        writer.writerow(["Average Answer Length", perf_metrics["average_answer_length"]])
        writer.writerow([])
        
        # Feedback Distribution
        writer.writerow(["Feedback Distribution"])
        writer.writerow(["Feedback Type", "Count"])
        for feedback_type, count in perf_metrics["feedback_distribution"].items():
            writer.writerow([feedback_type, count])
        writer.writerow([])
        
        # Topic Performance
        writer.writerow(["Topic Performance"])
        writer.writerow(["Topic", "Success Rate", "Correct", "Total"])
        for topic, metrics in sorted(perf_metrics["topic_performance"].items()):
            writer.writerow([
                topic,
                metrics.get("success_rate", "N/A"),
                metrics.get("correct", 0),
                metrics.get("total", 0)
            ])
        writer.writerow([])
        
        # Difficulty Performance
        writer.writerow(["Difficulty Performance"])
        writer.writerow(["Difficulty", "Success Rate", "Correct", "Total"])
        for diff, metrics in perf_metrics["difficulty_performance"].items():
            writer.writerow([
                diff,
                metrics.get("success_rate", "N/A"),
                metrics.get("correct", 0),
                metrics.get("total", 0)
            ])
        
        return output.getvalue()

    def export_analytics_to_json(self, analytics_data: Dict[str, Any], pretty: bool = True) -> str:
        """
        Exports analytics data to JSON format.
        
        Args:
            analytics_data: The analytics data dictionary from generate_analytics()
            pretty: Whether to format the JSON with indentation
            
        Returns:
            str: JSON formatted string containing the analytics data
        """
        import json
        from datetime import datetime
        
        # Add export metadata
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "analytics": analytics_data
        }
        
        if pretty:
            return json.dumps(export_data, indent=2)
        return json.dumps(export_data)

    def export_analytics(self, format: str = "csv", **kwargs) -> str:
        """
        Exports analytics data in the specified format.
        
        Args:
            format: The export format ("csv" or "json")
            **kwargs: Additional arguments for the specific export method
            
        Returns:
            str: The exported analytics data in the specified format
            
        Raises:
            ValueError: If an unsupported format is specified
        """
        analytics_data = self.generate_analytics()
        
        if format.lower() == "csv":
            return self.export_analytics_to_csv(analytics_data)
        elif format.lower() == "json":
            return self.export_analytics_to_json(analytics_data, **kwargs)
        else:
            raise ValueError(f"Unsupported export format: {format}") 