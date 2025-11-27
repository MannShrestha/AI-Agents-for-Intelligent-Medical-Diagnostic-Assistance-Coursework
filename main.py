import os 
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.ai_agent.utils.agents import Endocrinologist, Cardiologist, Pulmonologist, Psychologist, MultidisciplinaryTeam



# Load .env automatically from project root
load_dotenv()

# Read medical report
report_path = r"Medical_Reports/Medical Report - Michael Johnson - Panic Attack Disorder.txt"
with open(report_path, "r") as file:
    medical_report = file.read()


# Initialize individual agents
agents = {
    "Endocrinologist": Endocrinologist(medical_report),
    "Cardiologist": Cardiologist(medical_report),
    "Pulmonologist": Pulmonologist(medical_report),
    "Psychologist": Psychologist(medical_report)
}


# Function to run agent and get result
def get_response(agent_name, agent):
    response = agent.run()
    return agent_name, response



# parallelism and concurrency -> ThreadPoolExecutor
# Run all agents concurrently
responses = {}
with ThreadPoolExecutor() as executor:
    futures = {executor.submit(get_response, name, agent): name for name, agent in agents.items()}
    for future in as_completed(futures):
        name, response = future.result()
        responses[name] = response


# Combine results in MultidisciplinaryTeam agent
team_agent = MultidisciplinaryTeam(
    endocrinologist_report = responses["Endocrinologist"],
    cardiologist_report = responses["Cardiologist"],
    pulmonologist_report = responses["Pulmonologist"],
    psychologist_report = responses["Psychologist"]
)


# Generate final diagnosis
final_diagnosis = team_agent.run()
final_output = "### Final Diagnosis\n\n" + final_diagnosis

# Save output to file
output_path = "results/final_diagnosis.txt"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    f.write(final_output)

print(f"Final diagnosis saved to {output_path}")

