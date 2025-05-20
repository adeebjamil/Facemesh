import React, { useRef, useEffect, useState, useCallback } from "react";
import "./App.css";
import * as tf from "@tensorflow/tfjs";
// Use the older facemesh model that's already installed
import * as facemesh from "@tensorflow-models/facemesh";
import Webcam from "react-webcam";

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [faceCount, setFaceCount] = useState(0);

  // Draw mesh function
  const drawMesh = (predictions, ctx) => {
    // Always set the current face count, even if it's zero
    setFaceCount(predictions.length);
    
    // Only draw if there are faces
    if (predictions.length > 0) {
      // Loop through each prediction
      predictions.forEach((prediction) => {
        // Get keypoints
        const keypoints = prediction.scaledMesh;
        
        // Draw dots
        for (let i = 0; i < keypoints.length; i++) {
          const x = keypoints[i][0];
          const y = keypoints[i][1];
          
          ctx.beginPath();
          ctx.arc(x, y, 1.5, 0, 3 * Math.PI);
          ctx.fillStyle = "#00FF00";
          ctx.fill();
        }
        
        // Draw triangles (triangulation connects dots with lines)
        const annotations = prediction.annotations;
        
        // Draw the lines
        ctx.strokeStyle = "#00FFFF";
        ctx.lineWidth = 1;
        
        // Draw contours
        Object.values(annotations).forEach((points) => {
          ctx.beginPath();
          for (let i = 0; i < points.length; i++) {
            const x = points[i][0];
            const y = points[i][1];
            
            if (i === 0) {
              ctx.moveTo(x, y);
            } else {
              ctx.lineTo(x, y);
            }
          }
          ctx.stroke();
        });
      });
    }
  };

  // Load the facemesh model
  const runFacemesh = useCallback(async () => {
    const net = await facemesh.load({
      inputResolution: { width: 640, height: 480 },
      scale: 0.8,
    });
    console.log("Facemesh model loaded");
    
    // Start detection loop
    setInterval(() => {
      detect(net);
    }, 100);
  }, []);

  // Detect function
  const detect = async (net) => {
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {
      // Get Video Properties
      const video = webcamRef.current.video;
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;

      // Set video width
      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;

      // Set canvas width
      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;

      // Make Detections
      const face = await net.estimateFaces(video);

      // Get canvas context
      const ctx = canvasRef.current.getContext("2d");
      
      // Clear canvas
      ctx.clearRect(0, 0, videoWidth, videoHeight);
      
      // Draw mesh
      requestAnimationFrame(() => drawMesh(face, ctx));
    }
  };

  useEffect(() => {
    runFacemesh();
  }, [runFacemesh]);

  return (
    <div className="App">
      <div className="face-counter">
        Faces Detected: {faceCount}
      </div>
      <header className="App-header">
        <Webcam
          ref={webcamRef}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zIndex: 9,
            width: 640,
            height: 480,
          }}
        />

        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zIndex: 10,
            width: 640,
            height: 480,
          }}
        />
      </header>
    </div>
  );
}

export default App;



















.App {
  text-align: center;
}

/* Add this styling for the face counter to appear in the top right corner */
.face-counter {
  position: absolute;
  top: 20px;
  right: 20px;
  background-color: rgba(0, 0, 0, 0.7);
  color: white;
  padding: 10px 15px;
  border-radius: 5px;
  font-size: 18px;
  font-weight: bold;
  z-index: 20; /* Higher than canvas and webcam to ensure visibility */
  margin: 0;
}

.App-logo {
  height: 40vmin;
  pointer-events: none;
}

@media (prefers-reduced-motion: no-preference) {
  .App-logo {
    animation: App-logo-spin infinite 20s linear;
  }
}

.App-header {
  background-color: #282c34;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  font-size: calc(10px + 2vmin);
  color: white;
}

.App-link {
  color: #61dafb;
}

@keyframes App-logo-spin {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}