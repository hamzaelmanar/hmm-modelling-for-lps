import hypersync
from hypersync import (
    LogSelection,
    LogField,
    BlockField,
    #FieldSelection,
    #TransactionField,
    HexOutput,
    #Decoder
)
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

# Mint : 
# 0x7a53080ba414158be7ec69b987b5fb7d07dee101fe85488f0853ae16239d0bde
# Burn : 
# 0x0c396cd989a39f4459b5fa1aed6a9a8dcdbc45908acfd67e028cd568da98982c
# Swap : 
# 0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67

usdc_weth_pool = {"chain_name": "eth", 
                  "contract_address": "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640",
                  "out_path": "usdc_weth_pool.parquet",
                  "topics": ["0x7a53080ba414158be7ec69b987b5fb7d07dee101fe85488f0853ae16239d0bde",
                             "0x0c396cd989a39f4459b5fa1aed6a9a8dcdbc45908acfd67e028cd568da98982c",
                             "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"]}

async def hypersync_indexer(chain_name : str, contract_address : str, out_path : str,topics : list):
    # Initialize client
    client = hypersync.HypersyncClient(
        hypersync.ClientConfig(
            url=f"https://{chain_name}.hypersync.xyz",
            bearer_token=os.getenv("HYPERSYNC_BEARER_TOKEN"),  # Set in .env file
        )
    )

    # Define field selection
    field_selection = hypersync.FieldSelection(
        block=[
            BlockField.NUMBER, 
            BlockField.TIMESTAMP
        ],
        log=[
            LogField.BLOCK_NUMBER,
            LogField.TRANSACTION_HASH,
            LogField.TRANSACTION_INDEX,
            LogField.LOG_INDEX,
            LogField.DATA,
            LogField.ADDRESS,
            LogField.TOPIC0, 
            LogField.TOPIC1,
            LogField.TOPIC2,
            LogField.TOPIC3,
        ],
    )

    # Define query for UNI transfers
    
    # define height for to block
    height = await client.get_height()
    query = hypersync.Query(
        from_block=0,
        to_block=height,
        field_selection=field_selection,
        logs=[
            LogSelection(
                address=[contract_address],  
                topics=[
                    [topics[0]] # Mint
                ]
            ),
            LogSelection(
                address=[contract_address],  
                topics=[
                    [topics[1]] # Burn
                ]
            ),
            LogSelection(
                address=[contract_address],  
                topics=[
                    [topics[2]] # Swap
                ]
            )
        ]
    )

    # Configure output
    config = hypersync.StreamConfig(
        hex_output=HexOutput.PREFIXED
    )

    # Collect data to a Parquet file
    print("Fetching logs...")
    result = await client.collect_parquet(out_path, query, config)
    print(f"Success. Processed blocks from {query.from_block} to {query.to_block}.")

asyncio.run(
    hypersync_indexer(
        usdc_weth_pool["chain_name"],
        usdc_weth_pool["contract_address"], 
        usdc_weth_pool["out_path"], 
        usdc_weth_pool["topics"]))

