from fastapi import FastAPI, Response, status
from app.services.Recommendation import Recommendation

def create_app():
    app = FastAPI()

    @app.get("/")
    async def read_root(response: Response):
        response.status_code = status.HTTP_204_NO_CONTENT
        return {"message": "Hello, FastAPI!"}

    @app.get("/suggestions/{user_id}")
    async def suggestions(user_id: str, response: Response):
        recommend = Recommendation()
        result = await recommend.recommend(user_id)
        response.status_code = status.HTTP_200_OK
        return {"message": "Hello, FastAPI!", "data": result}

    @app.get("/test")
    async def possible_matches(response: Response):
        response.status_code = status.HTTP_200_OK
        # result = await Recommendation.possible_matches("68cdc013137f27f7eca9cd8f",50,20)
        result = await Recommendation.get_all_users()
        return {"message": result}

    return app
