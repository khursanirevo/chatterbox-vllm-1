#!/usr/bin/env python3
"""Test WebSocket connection directly."""
import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://localhost:8000/tts/websocket"
    print(f"Connecting to {uri}...")

    try:
        async with websockets.connect(uri) as ws:
            print("✅ Connected successfully!")

            # Send a test request
            request = {
                "text": "Hello world",
                "request_id": "test-connection"
            }
            await ws.send(json.dumps(request))
            print("📤 Sent request")

            # Wait for response
            response = await asyncio.wait_for(ws.recv(), timeout=10)
            data = json.loads(response)
            print(f"📥 Received: {data['type']}")

            # Receive more messages
            async for msg in ws:
                data = json.loads(msg)
                print(f"📥 Received: {data['type']}")
                if data['type'] in ['complete', 'error']:
                    break

    except websockets.exceptions.InvalidStatusCode as e:
        print(f"❌ Invalid status code: {e}")
        print("Server might not be running or CORS is blocking")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())
