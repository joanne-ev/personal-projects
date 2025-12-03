-- Import CSV file into a temp table
.import --csv Housing.csv housing_temp

-- Create new housing table
CREATE TABLE "housing" (
    "id" INTEGER PRIMARY KEY,
    "price" INTEGER,
    "area" INTEGER,
    "bedrooms" INTEGER,
    "bathrooms" INTEGER,
    "stories" INTEGER,
    "mainroad" TEXT,
    "guestroom" TEXT,
    "basement" TEXT,
    "hotwaterheating" TEXT,
    "airconditioning" TEXT,
    "parking" INTEGER,
    "prefarea" TEXT,
    "furnishingstatus" TEXT,

    CHECK (
        "mainroad" IN ('yes', 'no') AND
        "guestroom" IN ('yes', 'no') AND
        "basement" IN ('yes', 'no') AND
        "hotwaterheating" IN ('yes', 'no') AND
        "airconditioning" IN ('yes', 'no') AND
        "prefarea" IN ('yes', 'no') AND
        "furnishingstatus" IN ('unfurnished', 'semi-furnished', 'furnished')
    )
);

-- Move the data from housing_temp into the final table
INSERT INTO housing (
    "price", "area", "bedrooms", "bathrooms", "stories", 
    "mainroad", "guestroom", "basement", "hotwaterheating", 
    "airconditioning", "parking", "prefarea", "furnishingstatus"
)
SELECT 
    "price", "area", "bedrooms", "bathrooms", "stories", 
    "mainroad", "guestroom", "basement", "hotwaterheating", 
    "airconditioning", "parking", "prefarea", "furnishingstatus"
FROM housing_temp;

-- Delete temporary table
DROP TABLE housing_temp;