# ADAS Client

`adas-client` is a real-time Advanced Driver Assistance System (ADAS) frontend application built with [Next.js](https://nextjs.org) and [React](https://react.dev). It serves as the visualization and sensor streaming interface for a YOLO-based ADAS navigation system.

## Features

-   **Real-time Sensor Streaming**: Captures and streams transmission of:
    -   **Camera Feed**: High-frequency video frames from the device's camera.
    -   **GPS Data**: Real-time geolocation coordinates.
-   **Live Visualization**: Renders inference results received from the backend server on an overlay canvas:
    -   **Lane Detection**: Visualizes detected lane lines.
    -   **Trajectory Planning**: Displays the projected path of the vehicle.
    -   **Control Commands**: Visualizes steering angle predictions.
-   **Efficient Communication**: Uses **MessagePack** over **WebSockets** for high-performance, low-latency binary data exchange.
-   **Modern Tech Stack**: Built with Next.js 16, React 19, and Tailwind CSS v4.

## Technology Stack

-   **Framework**: [Next.js 16](https://nextjs.org/) (App Router)
-   **UI Library**: [React 19](https://react.dev/)
-   **Styling**: [Tailwind CSS v4](https://tailwindcss.com/)
-   **Protocol**: [MessagePack](https://msgpack.org/) (via `@msgpack/msgpack`)
-   **Networking**: Native WebSockets

## Project Structure

```bash
src/
├── app/             # App Router pages and layouts
├── components/      # UI components (e.g., OverlayCanvas)
├── hooks/           # Custom hooks for sensors and sockets
│   ├── useCamera.ts      # Camera stream management
│   ├── useGPS.ts         # Geolocation tracking
│   ├── useWebSocket.ts   # WebSocket connection handling
│   └── useSensorStream.ts # Orchestrates data streaming
├── lib/             # Core libraries (e.g., WebSocket codec)
└── utils/           # Helper functions (e.g., canvas drawing)
```

## Getting Started

### Prerequisites

-   Node.js 20+
-   npm, yarn, or pnpm

### Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd adas-client
    ```

2.  Install dependencies:
    ```bash
    npm install
    # or
    yarn install
    ```

### Development

To start the development server:

```bash
npm run dev
# or
yarn dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the application.

> **Note**: The application attempts to connect to a WebSocket server at `wss://192.168.1.7:8000/ws` by default. Ensure your backend server is running and accessible, or update the URL in `src/app/page.tsx` as needed.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
