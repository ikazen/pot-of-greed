from __future__ import annotations

import os

import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer


@cl.data_layer
def get_data_layer():
    return SQLAlchemyDataLayer(conninfo=os.environ["CHAINLIT_DB_DSN"])
