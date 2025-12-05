from pymongo import MongoClient
from pymongo.database import Database

from decouple import Config, RepositoryEnv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

config = Config(RepositoryEnv('.env'))

MONGO_HOST = config('MONGO_HOST', default='localhost:27017', cast=str)
MONGO_USER = config('MONGO_USER', default='', cast=str)
MONGO_PASS = config('MONGO_PASS', default='', cast=str)

CACHE_SIZE_MB = config('CACHE_SIZE_MB', default=1024, cast=int)


mongo = MongoClient(
    MONGO_HOST,
    username=MONGO_USER,
    password=MONGO_PASS,
)

db = mongo['wikidata']


def create_capped_collection(db: Database, name: str, size_mb: int):
    """
    Ensures that a capped collection with the given name exists.
    If the collection exists but is not capped, raises an error.

    :param db: pymongo Database instance
    :param name: collection name
    :param size_mb: maximum size in MB
    """

    size_bytes = size_mb * 1024 * 1024

    # Check if collection exists
    if name in db.list_collection_names():
        info = db.command("listCollections", filter={"name": name})
        options = info["cursor"]["firstBatch"][0].get("options", {})

        if not options.get("capped", False):
            logger.error(f"Collection '{name}' exists but is NOT capped.")
            raise RuntimeError(
                f"Collection '{name}' exists but is NOT capped. "
                "Please drop it manually or rename it."
            )
        else:
            # Already capped — nothing to do
            return

    # Create capped collection
    db.create_collection(
        name,
        capped=True,
        size=size_bytes
    )

    logger.info(f"Capped collection '{name}' created with size {size_mb} MB")


if __name__ == "__main__":
   create_capped_collection(db, 'cache', size_mb=CACHE_SIZE_MB)