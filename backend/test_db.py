import asyncio
from database import tickets_collection

async def test():
    #insert test document
    test_doc = {
        "issue": "test MongoDB connection",
        "system": "none",
        "impact": "none"
    }

    result = await tickets_collection.insert_one(test_doc)
    print(f"Inserted ID: {result.inserted_id}")

    #read it back
    doc = await tickets_collection.find_one({"issue": "test MongoDB connection"})
    print(f"Found: {doc}")

if __name__ == "__main__":
    asyncio.run(test())