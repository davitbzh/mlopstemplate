import hopsworks
from mlopstemplate.features import profile
from mlopstemplate.synthetic_data.data_sources import get_datasets

# get data from the source
_, _, profiles_df = get_datasets()

# compute profile features
# select final features
profiles_df = profile.select_features(profiles_df)

# connect to hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# get or create feature group
profile_fg = fs.get_or_create_feature_group(
    name="profile",
    version=1,
    description="Credit card holder demographic data",
    primary_key=["cc_num"],
    partition_key=["sex"],
    stream=True,
    online_enabled=True
)

# materialize feature data in to the feature group
profile_fg.insert(profiles_df)
