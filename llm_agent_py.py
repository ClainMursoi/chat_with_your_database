import os
from dotenv import load_dotenv
import json
import google.generativeai as genai
from database_py import DatabaseManager
from ml_agent import MLAgent

load_dotenv()
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

class SQLAgent:
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        self.ml_agent = MLAgent(self.db_manager, self.model)

    def get_database_context(self):
        schema, error = self.db_manager.get_table_schema()
        if error:
            return f"Error getting schema: {error}"
        
        context = "Database Schema Information:\n\n"
        for table_name, columns in schema.items():
            context += f"Table: {table_name}\nColumns:\n"
            for col in columns:
                context += f"  - {col['column']} ({col['type']})\n"
            context += "\n"
        return context

    def detect_intent(self, user_question):
        """Strong intent detection with keyword fallback"""
        lower_question = user_question.lower()

        # Strong keyword-based fallback for predictive questions
        predictive_keywords = ["predict", "forecast", "estimate future", "next years", "future trend", "what will be"]
        if any(keyword in lower_question for keyword in predictive_keywords):
            # Try to extract target and county
            target = "population"  # default
            county = None
            horizon = 5

            # Simple extraction
            if "population" in lower_question:
                target = "population"
            elif "enrolment" in lower_question or "enrollment" in lower_question:
                target = "enrolment"
            elif "gdp" in lower_question:
                target = "gdp_estimate"

            # Extract county name (simple approach)
            county_list = ["nakuru", "nairobi", "kisumu", "mombasa", "eldoret"]
            for c in county_list:
                if c in lower_question:
                    county = c.capitalize()
                    break

            return {
                "mode": "predictive",
                "params": {"target": target, "county": county, "horizon": horizon},
                "reason": "keyword fallback"
            }

        # Normal LLM-based detection as backup
        prompt = f"""
You MUST respond with valid JSON only.

Query: "{user_question}"

Output exactly:
{{
  "mode": "sql" or "predictive",
  "params": {{ "target": "population", "county": "Nakuru", "horizon": 5 }},
  "reason": "brief reason"
}}
"""

        try:
            response = self.model.generate_content(prompt)
            raw = response.text.strip()
            raw = raw.replace('```json', '').replace('```', '').strip()
            start = raw.find('{')
            end = raw.rfind('}') + 1
            cleaned = raw[start:end] if start != -1 else raw
            return json.loads(cleaned)
        except:
            return {"mode": "sql", "params": {}, "reason": "fallback to sql"}

    def generate_sql(self, user_question):
        prompt = f"""
You are a SQL expert. Generate ONLY a safe PostgreSQL SELECT query.

{self.get_database_context()}

Question: {user_question}

Return only the SQL query.
"""
        try:
            response = self.model.generate_content(prompt)
            sql = response.text.strip().replace('```sql', '').replace('```', '').strip()
            return sql, None
        except Exception as e:
            return None, str(e)

    def format_results_to_natural_language(self, results, question):
        if not results:
            return "No results found."
        prompt = f"Convert these results into natural language for question: {question}\nResults: {json.dumps(results, default=str)}"
        try:
            return self.model.generate_content(prompt).text.strip()
        except:
            return "Here are the results."

    def process_question(self, user_question):
        intent = self.detect_intent(user_question)
        mode = intent.get("mode", "sql")
        params = intent.get("params", {})

        if mode == "sql":
            sql_query, err = self.generate_sql(user_question)
            if err:
                return None, None, err
            results, db_err = self.db_manager.execute_query(sql_query)
            if db_err:
                return sql_query, None, db_err
            natural = self.format_results_to_natural_language(results, user_question)
            return sql_query, natural, None

        else:
            try:
                response = self.ml_agent.process_predictive(user_question, params)
                return None, response, None
            except ValueError as e:
                return None, None, str(e)