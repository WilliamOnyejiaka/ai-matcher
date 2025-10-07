from fastapi import FastAPI, Response, status


def create_app():
    app = FastAPI()

    @app.get("/")
    async def read_root(response: Response):
        response.status_code = status.HTTP_204_NO_CONTENT
        return {"message": "Hello, FastAPI!"}

    return app
