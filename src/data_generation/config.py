from pydantic import BaseModel, Field

class Config(BaseModel):
    #Configuration class for data generation.
    
    num_samples: int = Field(1000, description="Number of samples to generate")
    num_features: int = Field(10, description="Number of features for each sample")
    random_seed: int = Field(42, description="Random seed for reproducibility")
    output_file: str = Field("generated_data.csv", description="Path to the output file for generated data")
    

    