
@app.websocket("/tts/websocket")
async def tts_websocket(websocket: WebSocket):
    # Get the parameters from the query string
    params = websocket.query_params
    params.get("api_key", None)
    params.get("version", "2025-12-09")

    await websocket.accept()
    voice_session = TTSWebsocketSession()
    try:
        while True:
            data = await websocket.receive_json()

            print(f"Received WebSocket data: {data}")

            async for chunk in voice_session.handle_tts_event(raw_data=data):
                await websocket.send_text(json.dumps(chunk))
    except WebSocketDisconnect as e:
        logging.info(f"WebSocket disconnected: {e}")
    except Exception as e:
        logging.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        # Gracefully close WebSocket if still connected
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close()
        except (RuntimeError, AttributeError):
            # Already closed or closing - ignore
            pass
