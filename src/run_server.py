#!/usr/bin/env python
import uvicorn


if __name__ == "__main__":
    uvicorn.run("inference:app", port=8080, host="0.0.0.0", log_level="info")
