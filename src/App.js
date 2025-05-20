import React, { useRef, useEffect, useState, useCallback } from "react";
import "./App.css";
import * as tf from "@tensorflow/tfjs";
// Use the older facemesh model that's already installed
import * as facemesh from "@tensorflow-models/facemesh";
// Import handpose model for finger detection
import * as handpose from "@tensorflow-models/handpose";
import Webcam from "react-webcam";

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [faceCount, setFaceCount] = useState(0);
  const [fingerCount, setFingerCount] = useState(0);
  const [handNet, setHandNet] = useState(null);

  // Draw mesh function (face)
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

  // Improved finger detection function
  const isFingerExtended = (landmarks, fingerBase, fingerTip, mcpJoint) => {
    const basePosition = landmarks[fingerBase]; // Wrist or palm point
    const mcpPosition = landmarks[mcpJoint];    // Knuckle joint
    const tipPosition = landmarks[fingerTip];   // Fingertip
    
    // Calculate angle between the joints
    const getAngle = (p1, p2, p3) => {
      const v1 = [p1[0] - p2[0], p1[1] - p2[1]];
      const v2 = [p3[0] - p2[0], p3[1] - p2[1]];
      
      // Normalize vectors
      const v1mag = Math.sqrt(v1[0] * v1[0] + v1[1] * v1[1]);
      const v2mag = Math.sqrt(v2[0] * v2[0] + v2[1] * v2[1]);
      
      const v1norm = [v1[0] / v1mag, v1[1] / v1mag];
      const v2norm = [v2[0] / v2mag, v2[1] / v2mag];
      
      // Dot product
      const dotProduct = v1norm[0] * v2norm[0] + v1norm[1] * v2norm[1];
      
      // Get angle in radians and convert to degrees
      return Math.acos(Math.min(Math.max(dotProduct, -1), 1)) * (180 / Math.PI);
    };
    
    // Calculate distance between tip and base to check if finger is extended
    const distance = Math.sqrt(
      Math.pow(tipPosition[0] - basePosition[0], 2) + 
      Math.pow(tipPosition[1] - basePosition[1], 2)
    );
    
    // Calculate distance between knuckle and base for normalization
    const handSize = Math.sqrt(
      Math.pow(mcpPosition[0] - basePosition[0], 2) + 
      Math.pow(mcpPosition[1] - basePosition[1], 2)
    );
    
    // Calculate angle to determine if finger is straight
    const angle = getAngle(basePosition, mcpPosition, tipPosition);
    
    // Different logic for thumb
    if (fingerBase === 0 && fingerTip === 4) {
      // Thumb is extended if the angle between base-knuckle-tip is greater than 35°
      // and the distance between tip and base is significant
      return angle > 35 && distance > handSize * 0.35;
    } else {
      // For other fingers, they're extended if they're relatively straight (angle > 120°)
      // and the distance from tip to base is significant
      return angle > 120 && distance > handSize * 0.6;
    }
  };

  // Updated finger counting function with improved detection
  const countExtendedFingers = (landmarks) => {
    let count = 0;
    let extendedFingers = [];
    
    // Check thumb (landmarks 0, 4, 1)
    if (isFingerExtended(landmarks, 0, 4, 1)) {
      count++;
      extendedFingers.push("thumb");
    }
    
    // Check index finger (landmarks 0, 8, 5)
    if (isFingerExtended(landmarks, 0, 8, 5)) {
      count++;
      extendedFingers.push("index");
    }
    
    // Check middle finger (landmarks 0, 12, 9)
    if (isFingerExtended(landmarks, 0, 12, 9)) {
      count++;
      extendedFingers.push("middle");
    }
    
    // Check ring finger (landmarks 0, 16, 13)
    if (isFingerExtended(landmarks, 0, 16, 13)) {
      count++;
      extendedFingers.push("ring");
    }
    
    // Check pinky finger (landmarks 0, 20, 17)
    if (isFingerExtended(landmarks, 0, 20, 17)) {
      count++;
      extendedFingers.push("pinky");
    }
    
    return { count, extendedFingers };
  };

  // Updated draw hand mesh function with visual feedback
  const drawHandMesh = (predictions, ctx) => {
    if (predictions.length > 0) {
      // Loop through each prediction
      predictions.forEach((prediction) => {
        const landmarks = prediction.landmarks;
        
        // Get extended fingers data
        const { count, extendedFingers } = countExtendedFingers(landmarks);
        
        // Update finger count with smoothing (prevents rapid fluctuations)
        if (Math.abs(count - fingerCount) > 0) {
          setFingerCount(count);
        }
        
        // Define connections between landmarks
        const connections = [
          // Thumb
          [0, 1], [1, 2], [2, 3], [3, 4],
          // Index finger
          [0, 5], [5, 6], [6, 7], [7, 8],
          // Middle finger
          [0, 9], [9, 10], [10, 11], [11, 12],
          // Ring finger
          [0, 13], [13, 14], [14, 15], [15, 16],
          // Pinky
          [0, 17], [17, 18], [18, 19], [19, 20],
          // Palm
          [0, 5], [5, 9], [9, 13], [13, 17]
        ];
        
        // Define finger groups for coloring
        const fingerGroups = {
          thumb: [1, 2, 3, 4],
          index: [5, 6, 7, 8],
          middle: [9, 10, 11, 12],
          ring: [13, 14, 15, 16],
          pinky: [17, 18, 19, 20]
        };
        
        // Draw landmarks
        for (let i = 0; i < landmarks.length; i++) {
          const [x, y] = landmarks[i];
          ctx.beginPath();
          
          // Bigger circles for fingertips
          const isFingertip = [4, 8, 12, 16, 20].includes(i);
          const radius = isFingertip ? 5 : 3;
          
          ctx.arc(x, y, radius, 0, 2 * Math.PI);
          
          // Color wrist/palm point differently
          if (i === 0) {
            ctx.fillStyle = "#FF0000"; // Red for palm center
          } else {
            // Check if this point belongs to an extended finger
            let isExtended = false;
            for (const finger of extendedFingers) {
              if (fingerGroups[finger].includes(i)) {
                isExtended = true;
                break;
              }
            }
            
            // Color based on whether the finger is extended
            ctx.fillStyle = isExtended ? "#00FF00" : "#FF9900";
          }
          
          ctx.fill();
        }
        
        // Draw connections (lines)
        for (let i = 0; i < connections.length; i++) {
          const [idx1, idx2] = connections[i];
          const pt1 = landmarks[idx1];
          const pt2 = landmarks[idx2];
          
          // Determine if this connection is part of an extended finger
          let isExtendedConnection = false;
          for (const finger of extendedFingers) {
            if (fingerGroups[finger].includes(idx1) && fingerGroups[finger].includes(idx2)) {
              isExtendedConnection = true;
              break;
            }
          }
          
          ctx.beginPath();
          ctx.moveTo(pt1[0], pt1[1]);
          ctx.lineTo(pt2[0], pt2[1]);
          ctx.strokeStyle = isExtendedConnection ? "#00FF00" : "#FF9900"; // Green for extended fingers
          ctx.lineWidth = 2;
          ctx.stroke();
        }
        
        // Add labels above fingertips for debugging
        ctx.font = '12px Arial';
        ctx.fillStyle = 'white';
        
        const fingertips = [
          { index: 4, name: "Thumb" },
          { index: 8, name: "Index" },
          { index: 12, name: "Middle" },
          { index: 16, name: "Ring" },
          { index: 20, name: "Pinky" }
        ];
        
        for (const tip of fingertips) {
          const [x, y] = landmarks[tip.index];
          const isExtended = extendedFingers.includes(tip.name.toLowerCase());
          ctx.fillText(`${tip.name}${isExtended ? " ✓" : ""}`, x, y - 10);
        }
      });
    } else {
      // Reset finger count when no hands are detected
      setFingerCount(0);
    }
  };
  
  // Load the facemesh and handpose models
  const runModels = useCallback(async () => {
    await tf.setBackend('webgl');
    
    // Load facemesh model
    const faceNet = await facemesh.load({
      inputResolution: { width: 640, height: 480 },
      scale: 0.8,
    });
    console.log("Facemesh model loaded");
    
    // Load handpose model
    const hand = await handpose.load();
    console.log("Handpose model loaded");
    setHandNet(hand);
    
    // Start detection loop
    setInterval(() => {
      detect(faceNet, hand);
    }, 100);
  }, []);

  // Detect function for both face and hand
  const detect = async (faceNet, handNet) => {
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {
      // Get Video Properties
      const video = webcamRef.current.video;
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;

      // Set video dimensions
      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;

      // Set canvas dimensions
      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;

      // Get canvas context
      const ctx = canvasRef.current.getContext("2d");
      
      // Clear canvas
      ctx.clearRect(0, 0, videoWidth, videoHeight);
      
      try {
        // Make Face Detections
        const faces = await faceNet.estimateFaces(video);
        drawMesh(faces, ctx);
        
        // Make Hand Detections
        if (handNet) {
          const hands = await handNet.estimateHands(video);
          drawHandMesh(hands, ctx);
        }
      } catch (error) {
        console.error("Error in detection:", error);
      }
    }
  };

  useEffect(() => {
    runModels();
  }, [runModels]);

  return (
    <div className="App">
      <div className="face-counter">
        Faces Detected: {faceCount}
      </div>
      <div className="finger-counter">
        Fingers Extended: {fingerCount}
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