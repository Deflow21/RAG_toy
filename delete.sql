DELETE
FROM abc_documents
WHERE model_id NOT IN (
    SELECT model_id
    FROM (
        SELECT model_id
        FROM (
            SELECT DISTINCT model_id, doc_type
            FROM abc_documents
        ) AS sub
        GROUP BY model_id
        HAVING COUNT(*) = 2
    ) AS x
);
