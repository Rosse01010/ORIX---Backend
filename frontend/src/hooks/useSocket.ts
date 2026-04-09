import { useEffect, useRef, useState } from "react";
import { getSocket } from "../services/socketService";
import type { OrixSocket } from "../types/Socket";

/**
 * React hook that exposes the shared socket plus its connected state.
 */
export function useSocket(): { socket: OrixSocket; connected: boolean } {
  const socketRef = useRef<OrixSocket>(getSocket());
  const [connected, setConnected] = useState<boolean>(
    socketRef.current.connected
  );

  useEffect(() => {
    const s = socketRef.current;
    const onConnect = () => setConnected(true);
    const onDisconnect = () => setConnected(false);

    s.on("connect", onConnect);
    s.on("disconnect", onDisconnect);

    if (!s.connected) s.connect();

    return () => {
      s.off("connect", onConnect);
      s.off("disconnect", onDisconnect);
    };
  }, []);

  return { socket: socketRef.current, connected };
}
