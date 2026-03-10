import pandas as pd

def to_uint_series(series: pd.Series) -> pd.Series:
    return pd.Series(
        [int(x, 16) if isinstance(x, str) and x else 0 for x in series],
        index=series.index
    )

def to_int256_series(series: pd.Series) -> pd.Series:
    uints = to_uint_series(series)
    return pd.Series(
        [x - 2**256 if x >= 2**255 else x for x in uints],
        index=series.index
    )

def decode_mint_events(pool_events):
    data_clean = pool_events["data"].str[2:]

    sender = "0x" + data_clean.str.slice(24, 64)
    amount = to_uint_series(data_clean.str.slice(64, 128))
    amount0 = to_uint_series(data_clean.str.slice(128, 192))
    amount1 = to_uint_series(data_clean.str.slice(192, 256))

    owner = "0x" + pool_events["topic1"].str[26:]
    tickLower = to_int256_series(pool_events["topic2"])
    tickUpper = to_int256_series(pool_events["topic3"])

    return pd.concat(
        [
            pool_events,
            pd.DataFrame(
            {
            'sender': sender,
            'owner': owner,
            'tickLower': tickLower,
            'tickUpper': tickUpper,
            'amount': amount,
            'amount0': amount0,
            'amount1': amount1
        }
        )],
        axis=1)

def decode_burn_events(pool_events):
    data_clean = pool_events["data"].str[2:]

    amount = to_uint_series(data_clean.str.slice(0, 64))
    amount0 = to_uint_series(data_clean.str.slice(64, 128))
    amount1 = to_uint_series(data_clean.str.slice(128, 192))

    owner = "0x" + pool_events["topic1"].str[26:]
    tickLower = to_int256_series(pool_events["topic2"])
    tickUpper = to_int256_series(pool_events["topic3"])

    return pd.concat(
    [
        pool_events, 
        pd.DataFrame(
        {
            "owner": owner,
            "tickLower": tickLower,
            "tickUpper": tickUpper,
            "amount": amount,
            "amount0": amount0,
            "amount1": amount1,
        }
    )],
        axis=1)

def decode_swap_events(pool_events):
    data_clean = pool_events["data"].str[2:]

    amount0 = to_int256_series(data_clean.str.slice(0, 64))
    amount1 = to_int256_series(data_clean.str.slice(64, 128))
    sqrtPriceX96 = to_uint_series(data_clean.str.slice(128, 192))
    liquidity = to_uint_series(data_clean.str.slice(192, 256))
    tick = to_int256_series(data_clean.str.slice(256, 320))

    sender = "0x" + pool_events["topic1"].str[26:]
    recipient = "0x" + pool_events["topic2"].str[26:]

    return pd.concat(
        [
            pool_events,
        pd.DataFrame({
            "sender": sender,
            "recipient": recipient,
            "amount0": amount0,
            "amount1": amount1,
            "sqrtPriceX96": sqrtPriceX96,
            "liquidity": liquidity,
            "tick": tick
        })
        ],
        axis=1,
    )