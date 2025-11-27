import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate


class Agent:
    def __init__(self, medical_report = None, role = None, extra_info = None):
        self.medical_report = medical_report
        self.role = role
        self.extra_info = extra_info

        self.prompt_template = self.create_prompt_template()

        ## Load Google API key from .env
        google_Key = os.getenv("GOOGLE_API_KEY")
        if not google_Key:
            raise ValueError("Google Api key not found in .env")
        
        ## Initialize Gemini 2.5 Flash model
        self.model = ChatGoogleGenerativeAI(
            model = "gemini-2.5-flash",
            temperature = 0,
            google_api_key = google_Key

        )

    ## Creating prompt template
    def create_prompt_template(self):
        if self.role == "MultidisciplinaryTeam":
            template = f"""
            Act like a multidisciplinary panel of medical specialists.
            You receive diagnostic reports from a Endocrinologist, Cardiologist, Pulmonologist and Psychologist.
            Analyze them together and provide 4 possible health issues affecting the patient with reasoning.

            Return ONLY 3 bullet points, each with:
            - Condition
            - Short justification

            --- Reports ---
            Endocrinologist: {self.extra_info.get('endocrinologist_report', '')}
            Cardiologist: {self.extra_info.get('cardiologist_report', '')}
            Pulmonologist: {self.extra_info.get('pulmonologist_report', '')}
            Psychologist: {self.extra_info.get('psychologist_report', '')}
            """
        else:
            template = {
            "Endocrinologist": """
                You are a endocrinologist reviewing a patient medical report.
                Identify any possible hormone-related causes and recommend next diagnostic steps.
                Only output:
                - Possible hormone-related Causes
                - Recommended Next Steps
                Medical Report: {medical_report}
            """,
            "Cardiologist": """
                You are a cardiologist reviewing a patient medical report.
                Identify any possible cardiac causes and recommend next diagnostic steps.
                Only output:
                - Possible Cardiac Causes
                - Recommended Next Steps
                Medical Report: {medical_report}
            """,
            "Pulmonologist": """
                You are a pulmonologist reviewing patient data.
                Identify possible respiratory issues and next steps.
                Output only:
                - Possible Pulmonary Issues
                - Recommended Testing / Management
                Report: {medical_report}
            """,
            "Psychologist": """
                You are a psychologist reviewing the patient medical report.
                Identify possible psychological concerns and recommend next steps.
                Output only:
                - Possible Mental Health Causes
                - Next Steps
                Report: {medical_report}
            """
            }[self.role]
        
        return PromptTemplate.from_template(template)
    
    def run(self):
        print(f"Running {self.role} agent...")
        prompt = self.prompt_template.format(medical_report = self.medical_report)

        try:
            response = self.model.invoke(prompt)
            return response.content if hasattr(response, "content") else response
        except Exception as e:
            print(f"Error in {self.role} agent:", e)
            return None


# define Agent Classes
class Endocrinologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Endocrinologist")

class Cardiologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Cardiologist")

class Pulmonologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Pulmonologist")

class Psychologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Psychologist")


class MultidisciplinaryTeam(Agent):
    def __init__(self, endocrinologist_report, cardiologist_report, pulmonologist_report, psychologist_report):
        extra_info = {
            "endocrinologist_report": endocrinologist_report,
            "cardiologist_report": cardiologist_report,
            "pulmonologist_report": pulmonologist_report,
            "psychologist_report": psychologist_report
        }
        super().__init__(role="MultidisciplinaryTeam", extra_info=extra_info)

