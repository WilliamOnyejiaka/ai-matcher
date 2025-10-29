from app.config.db import db
from bson import ObjectId
from .AI import AI


class Recommendation:

    user_collection = db["users"]
    ai = AI()

    @classmethod
    async def get_all_users(cls, batch_size: int = 100):
        cursor = cls.user_collection.find(
            {}, {"_id": 1, "location": 1, "gender": 1}).batch_size(batch_size)
        return [dict(user, _id=str(user['_id'])) for user in await cursor.to_list()]

    @classmethod
    async def possible_matches_location(cls, user_id: str, max_distance_km: int, limit: int):
        user = await cls.user_collection.find_one({'_id': ObjectId(user_id)})
        coordinates = user.get("location", {}).get("coordinates")
        gender = user.get("gender", {})
        batchSize = limit * 3
        page = 1
        skip = (page - 1) * batchSize


        pipeline = [
            {
                "$geoNear": {
                    "near": {"type": "Point", "coordinates": coordinates},
                    # Include distance in output (in meters)
                    "distanceField": "distance",
                    "maxDistance": max_distance_km * 1000,  # Convert km to meters
                    "spherical": True,  # Required for 2dsphere index
                    "query": {
                        "status": "active",
                        "_id": {"$ne": ObjectId(user_id)},
                        # Add gender filter if provided
                        **({"genderInterest": gender} if gender else {})
                    }
                }
            },
            {
                "$lookup": {
                    "from": "likehistories",
                    "let": {"userId": "$_id"},
                    "pipeline": [
                        {
                            "$match": {
                                "$expr": {
                                    "$and": [
                                        # Reference user who liked
                                        {"$eq": ["$userId",
                                                 ObjectId(user_id)]},
                                        # User from outer query
                                        {"$eq": ["$likedUserId", "$$userId"]}
                                    ]
                                }
                            }
                        }
                    ],
                    "as": "likeHistory"
                }
            },
            # Exclude users with matching like history
            {"$match": {"likeHistory": {"$size": 0}}},
            {"$project": {"_id": 1}},
            {"$limit": batchSize},
            {"$sample": {"size": skip + batchSize}}

        ]

        cursor = cls.user_collection.aggregate(pipeline)
        return [dict(user, _id=str(user['_id'])) for user in await cursor.to_list()]
    
    @classmethod
    async def possible_matches(cls, user_id: str, limit: int):
        user = await cls.user_collection.find_one({'_id': ObjectId(user_id)})
        gender = user.get("gender", {})
        batchSize = limit * 3
        page = 1
        skip = (page - 1) * batchSize

        pipeline = [
            {
                "$match": {
                    "status": "active",
                    "_id": {"$ne": ObjectId(user_id)},
                    **({"genderInterest": gender} if gender else {})
                }
            },
            {
                "$lookup": {
                    "from": "likehistories",
                    "let": {"userId": "$_id"},
                    "pipeline": [
                        {
                            "$match": {
                                "$expr": {
                                    "$and": [
                                        # Reference user who liked
                                        {"$eq": ["$userId",
                                                 ObjectId(user_id)]},
                                        # User from outer query
                                        {"$eq": ["$likedUserId", "$$userId"]}
                                    ]
                                }
                            }
                        }
                    ],
                    "as": "likeHistory"
                }
            },
            # Exclude users with matching like history
            {"$match": {"likeHistory": {"$size": 0}}},
            {"$project": {
                "_id": 1,
                "firstName": {"$ifNull": ["$firstName", None]},
                "lastName":  {"$ifNull": ["$lastName",  None]},
                "gender": 1,
                "dateOfBirth": 1,
                "height": 1,
                "photo": {"$ifNull": ["$photo", None]},
                "location": 1,
                "age": 1,                     # already calculated field?
                "score": 1,
                "hobbies": 1,
                "interests": 1,
                "pets": "$pets",
                "favoriteColors": 1,
                "spokenLanguages": 1,
                "embedding": 1
            }},
            {"$limit": batchSize},
            {"$sample": {"size": skip + batchSize}}

        ]

        cursor = cls.user_collection.aggregate(pipeline)
        user['_id'] = str(user['_id'])
        return {
            "user": user,
            "matches": [dict(user, _id=str(user['_id'])) for user in await cursor.to_list()]
        }

    @classmethod
    async def recommend(cls,user_id:str, limit: int=20):
        result = await cls.possible_matches(user_id, limit)
        matches = result['matches']
        query_embedding_arr = result['user']['embedding']

        return cls.ai.search(query_embedding_arr, matches)

    @classmethod
    async def compute_suggestions(cls, batch_size=100):
        cursor = cls.user_collection.find(
            {}, {"_id": 1, "location": 1, "gender": 1}).batch_size(batch_size)
        
        async for user in cursor:
            print(user)
        # return [dict(user, _id=str(user['_id'])) for user in await cursor.to_list()]
