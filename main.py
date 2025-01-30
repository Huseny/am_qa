from fastapi import FastAPI
from pydantic import BaseModel
from simpletransformers.question_answering import QuestionAnsweringModel
import torch

app = FastAPI()


class QARequest(BaseModel):
    context: str
    question: str


MODEL_TYPE = "xlmroberta"
MODEL_PATH = f"Huseny/amha_qa"

model = QuestionAnsweringModel(
    MODEL_TYPE,
    MODEL_PATH,
    use_cuda=torch.cuda.is_available(),
    args={"silent": True},  # Suppress extra logging
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
def predict_qa(request: QARequest):
    to_predict = [
        {
            "context": request.context,
            "qas": [{"question": request.question, "id": "112343"}],
        }
    ]
    try:
        answers, _ = model.predict(to_predict)
        print(answers)
        if answers and len(answers[0]["answer"]) > 0:
            return {"answer": answers[0]["answer"]}
        else:
            return {"error": "No valid answer found"}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)
