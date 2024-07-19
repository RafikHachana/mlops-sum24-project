import os
import great_expectations as gx
from great_expectations.data_context.types.base import DataContextConfig, FilesystemStoreBackendDefaults

# Define the path for the data context
data_context_path = "services/gx"

# Set up the data context configuration
context_config = DataContextConfig(
    store_backend_defaults=FilesystemStoreBackendDefaults(root_directory=data_context_path)
)

# Initialize the data context
context = gx.data_context.DataContext(config=context_config)

# Configure the data source
datasource_config = {
    "name": "my_datasource",
    "class_name": "Datasource",
    "execution_engine": {
        "class_name": "PandasExecutionEngine"
    },
    "data_connectors": {
        "default_inferred_data_connector_name": {
            "class_name": "InferredAssetFilesystemDataConnector",
            "base_directory": "data",
            "default_regex": {
                "group_names": ["data_asset_name"],
                "pattern": "(.*)"
            }
        }
    }
}

context.add_datasource(**datasource_config)

# Save the context for later use
context.save()

# Export context for other scripts
def get_context():
    return context
