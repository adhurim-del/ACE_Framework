import asyncio
import logging
import aio_pika
from abc import ABC, abstractmethod
from base.settings import Settings
import base.prompts as p
from base.amqp.connection import get_connection
from base.amqp.exchange import create_exchange
import time
import openai
from typing import List
import re
import tiktoken


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseLayer(ABC):

    def __init__(self, settings: Settings):
        self.settings = settings
        self.loop = asyncio.get_event_loop()
        self.connection = None
        self.channel = None
        self.memory = []

        logger.info(f"setting {self.settings.role_name} Layer's mission")
        self._generate_completion(
            self.get_primary_directive(),
            role="system",
        )
        identity_summary = self._generate_completion(p.who_are_you)
        logger.info(identity_summary)

    @abstractmethod
    def get_primary_directive(self):
        pass

    # Override this function in your subclass, the default behavior here is to confirm the system is up and running
    async def northbound_message_handler(self, message: aio_pika.IncomingMessage):
        msg = message.body.decode()
        logger.info(f"received message = {msg}")
        await self._handle_bus_message(message=msg, source="Data Bus Message")
        await message.ack()

    # Override this function in your subclass, the default behavior here is to confirm the system is up and running
    async def southbound_message_handler(self, message: aio_pika.IncomingMessage):
        msg = message.body.decode()
        logger.info(f"received message = {msg}")
        await self._handle_bus_message(message=msg, source="Control Bus Message")
        await message.ack()

    async def _handle_bus_message(self, message, source):
        data_bus_prompt = p.DefaultMessageHandlerPrompt(
            source=source,
            message=message,
            layer=self.settings.role_name,
            destination="Data Bus Message",
        ).generate_prompt()

        control_bus_prompt = p.DefaultMessageHandlerPrompt(
            source=source,
            message=message,
            layer=self.settings.role_name,
            destination="Control Bus Message",
        ).generate_prompt()

        data_bus_message = self._generate_completion(data_bus_prompt)
        control_bus_message = self._generate_completion(control_bus_prompt)

        logger.info(f"{data_bus_message=}")
        logger.info(f"{control_bus_message=}")

        await self._publish(
            queue_name=self.settings.northbound_publish_queue,
            message=data_bus_message,
        )
        await self._publish(
            queue_name=self.settings.southbound_publish_queue,
            message=control_bus_message,
        )

    async def _connect(self):
        self.connection = await get_connection(
            loop=self.loop,
            amqp_host_name=self.settings.amqp_host_name,
            username=self.settings.amqp_username,
            password=self.settings.amqp_password,
            role_name=self.settings.role_name,
        )
        self.channel = await self.connection.channel()
        logger.info(f"{self.settings.role_name} connection established...")

    async def _subscribe(self):
        nb_queue = await self.channel.declare_queue(self.settings.northbound_subscribe_queue, durable=True)
        sb_queue = await self.channel.declare_queue(self.settings.southbound_subscribe_queue, durable=True)

        await nb_queue.consume(self.northbound_message_handler)
        await sb_queue.consume(self.southbound_message_handler)

    def _generate_completion(self, new_message, role="user"):
        openai.api_key = self.settings.openai_api_key
        primary_directive = {"role": "system", "content": self.get_primary_directive()}
        new_prompt = {"role": role, "content": new_message}
        conversation = (
            [primary_directive] +
            self.memory +
            [new_prompt]
        )
        completion = openai.ChatCompletion.create(
            model=self.settings.model,
            messages=conversation,
            temperature=self.settings.temperature,
        )
        response = completion.choices[0].message

        if role in ["user", "assistant"]:
            self.memory.append(new_prompt)
            self.memory.append(response)
            self._compact_memory()

        return response["content"]

    def _compact_memory(self):
        token_count = 0
        for message in self.memory:
            token_count += self._count_tokens(message)

        if token_count > self.settings.memory_max_tokens:
            logger.info("memory {token_count=}, compacting initiated...")

            self._update_memory()

        logger.info(f"{self.memory=}")
        token_count = self._count_tokens(self.memory[0])
        logger.info("after compaction memory {token_count=}")

    def _update_memory(self):
        openai.api_key = self.settings.openai_api_key
        primary_directive = {"role": "system", "content": self.get_primary_directive()}
        summarization_prompt = {"role": "user", "content": p.memory_compaction_prompt}
        conversation = (
                [primary_directive] +
                self.memory +
                [summarization_prompt]
            )
        completion = openai.ChatCompletion.create(
                model=self.settings.model,
                messages=conversation,
                temperature=self.settings.temperature,
            )
        self.memory = [completion.choices[0].message]

    def _count_tokens(self, message: str) -> int:
        encoding  = tiktoken.encoding_for_model(self.settings.model)

        logger.info(f"{message=}")

        num_tokens = len(encoding.encode(message["content"]))
        return num_tokens

    async def _publish(self, queue_name, message):
        if self._determine_none(message) != 'none':
            exchange = await create_exchange(
                connection=self.connection,
                queue_name=queue_name,
            )
            message_body = aio_pika.Message(
                body=message.encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT
            )
            await exchange.publish(
                message_body,
                routing_key=queue_name,
            )

    def _determine_none(self, input_text):
        match = re.search(r'\[Message\]\n(none)', input_text)

        if match:
            return 'none'

        return input_text

    async def _run_layer(self):
        logger.info(f"Running {self.settings.role_name}")
        await self._connect()
        await self._subscribe()
        logger.info(f"{self.settings.role_name} Subscribed to {self.settings.northbound_subscribe_queue} and {self.settings.southbound_subscribe_queue}")

    def run(self):
        self.loop.create_task(self._run_layer())
        try:
            self.loop.run_forever()
        finally:
            self.loop.close()


# def closest_match(input_str: str, options: List[str]):
#     scores = [fuzz.ratio(input_str, option) for option in options]
#     return options[scores.index(max(scores))]

