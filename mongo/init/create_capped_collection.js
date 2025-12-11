db = db.getSiblingDB("mydb");

db.createCollection("cache", {
    capped: true,
    size: 2048 * 1024 * 1024,   // 2 GB
    max: 0                      // unlimited docs, limited by size
});