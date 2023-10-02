from base.settings import Settings


settings = Settings(
    role_name="Executive Layer",
    northbound_subscribe_queue="northbound.layer_4_executive",
    southbound_subscribe_queue="southbound.layer_4_executive",
    southbound_publish_queue="southbound.layer_5_controller",
    northbound_publish_queue="northbound.layer_3_agent",

)
