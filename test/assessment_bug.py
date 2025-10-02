import mlflow
from mlflow.entities import Expectation, ExpectationValue

# Create an expectation with value 42.
response = mlflow.log_assessment(
    trace_id="1234",
    assessment=Expectation(name="expected_answer", value=42),
)
assessment_id = response.assessment_id

# Update the expectation with a new value 43.
mlflow.update_assessment(
    trace_id="1234",
    assessment_id=assessment_id.assessment_id,
    assessment=Expectation(name="expected_answer", value=43),
)
