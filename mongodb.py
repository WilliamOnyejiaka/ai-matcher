from typing import List
from app.config.db import db
import asyncio
from bson import ObjectId


async def test():
    try:
        collection = db["users"]
        # Use async for to iterate over the cursor
        async for user in collection.find({}).limit(2):
            print(user['_id'])
    except Exception as e:
        print(f"Error querying MongoDB: {e}")


async def possible_matches(user_id: str, max_distance_km: int):
    collection = db["users"]

    user = await collection.find_one({'_id': ObjectId(user_id)})
    coordinates = user.get("location", {}).get("coordinates")
    gender = user.get("gender", {})

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
        {"$limit": 100}  # Limit to 100 results for efficiency
    ]

    cursor = collection.aggregate(pipeline)
    return await cursor.to_list()

# Run the async function
if __name__ == "__main__":
    # asyncio.run(test())
    asyncio.run(possible_matches("68cdc013137f27f7eca9cd8f", 1))
