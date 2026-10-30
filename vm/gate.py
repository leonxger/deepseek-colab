"""TCP forwarder started only after READY, so port 18080 exists only once every weight byte is in place. usage: python3 gate.py LISTEN_PORT TARGET_PORT"""
import asyncio, socket, sys

LISTEN, TARGET = int(sys.argv[1]), int(sys.argv[2])


def nodelay(w):
    s = w.get_extra_info('socket')
    if s is not None:
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


async def pipe(r, w):
    try:
        while True:
            b = await r.read(65536)
            if not b:
                break
            w.write(b)
            await w.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        try:
            w.close()
        except Exception:
            pass


async def handle(cr, cw):
    try:
        ur, uw = await asyncio.open_connection('127.0.0.1', TARGET)
    except OSError:
        cw.close()
        return
    nodelay(cw)
    nodelay(uw)
    await asyncio.gather(pipe(cr, uw), pipe(ur, cw))


async def main():
    srv = await asyncio.start_server(handle, '127.0.0.1', LISTEN, reuse_address=True)
    print('gate open %d -> %d' % (LISTEN, TARGET), flush=True)
    async with srv:
        await srv.serve_forever()


asyncio.run(main())
