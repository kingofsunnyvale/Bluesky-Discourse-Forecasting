from openai import OpenAI

client = OpenAI(
  api_key="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",
  base_url="https://integrate.api.nvidia.com/v1"
)

response = client.embeddings.create(
    input=["What is the capital of France?"],
    model="nvidia/nv-embed-v1",
    encoding_format="float",
    extra_body={"input_type": "query", "truncate": "NONE"}
)

print(response.data[0].embedding)