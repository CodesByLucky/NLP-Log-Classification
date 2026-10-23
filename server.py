"""FastAPI service: upload a CSV of logs, get it back with a target_label column."""
import io

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from classify import classify_dataframe

REQUIRED_COLUMNS = {"source", "log_message"}

app = FastAPI(
    title="Hybrid Log Classification API",
    description="Classifies log messages with regex rules, sentence embeddings + Logistic Regression, and an LLM.",
    version="1.1.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/classify/")
async def classify_logs(file: UploadFile):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV.")

    try:
        df = pd.read_csv(file.file, skipinitialspace=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse the CSV file.")
    finally:
        file.file.close()

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"CSV must contain 'source' and 'log_message' columns (missing: {', '.join(sorted(missing))}).",
        )

    try:
        result = classify_dataframe(df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Classification failed: {exc}")

    # Build the response in memory so concurrent requests never share a file.
    buffer = io.StringIO()
    result.to_csv(buffer, index=False)
    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="classified_logs.csv"'},
    )
