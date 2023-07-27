import re

def df_schema_to_udf_schema(spark_df):
    """
        returns the dataframe schema in string format
    """
    return ", ".join([f"{field.name} {field.dataType.simpleString()}" for field in spark_df.schema.fields])


def add_feature_to_udf_schema(udf_schema, feature_name, feature_type):
    """
        add the new feature to the schema.
        if the feature already exists, ignore.
    """
    if not re.findall(r'\b' + feature_name + r'\b', udf_schema, flags=re.IGNORECASE):
        return udf_schema + ", " + feature_name + " " + feature_type
    else:
        return udf_schema
