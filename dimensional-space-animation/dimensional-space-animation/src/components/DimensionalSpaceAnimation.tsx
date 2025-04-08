import React, { useState, useEffect } from 'react';

interface StepProps {
  step: number;
}

interface AnimationControlsProps {
  step: number;
  totalSteps: number;
  autoplay: boolean;
  onStepChange: (step: number) => void;
  onAutoplayToggle: () => void;
}

// Global styles for animations
const globalStyles = `
  @keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.1); }
    100% { transform: scale(1); }
  }
  
  @keyframes float {
    0% { transform: translate(0, 0); }
    50% { transform: translate(5px, 5px); }
    100% { transform: translate(0, 0); }
  }
`;

const MathematicalRepresentation: React.FC<StepProps> = ({ step }) => {
  return (
    <div className="flex flex-col items-center justify-center p-4">
      <h2 className="text-2xl font-bold mb-8">Mathematical Representation of 64D Space</h2>
      <div className="w-full max-w-2xl bg-blue-50 rounded-lg p-6 shadow-md">
        <div className="text-lg mb-4">A 64-dimensional vector is simply an ordered list of 64 numbers:</div>
        <div className="font-mono text-sm bg-white p-3 rounded overflow-x-auto">
          <span className="text-blue-600">v = (</span>
          <span className="text-red-500">v₁, v₂, v₃, ..., v₆₄</span>
          <span className="text-blue-600">)</span>
        </div>
        <div className="mt-8 text-lg">Each number represents a value along one of the 64 dimensions</div>
        <div className="flex justify-center mt-8">
          <div className="relative h-40 w-40">
            <div className="absolute top-0 left-0 w-full h-full flex items-center justify-center">
              <div className="bg-blue-100 w-32 h-32 rounded-full flex items-center justify-center">
                <div className="bg-blue-200 w-24 h-24 rounded-full flex items-center justify-center">
                  <div className="bg-blue-300 w-16 h-16 rounded-full flex items-center justify-center">
                    <div className="bg-blue-400 w-8 h-8 rounded-full"></div>
                  </div>
                </div>
              </div>
            </div>
            <div className="absolute top-1/2 left-0 w-full h-1 bg-red-400 transform -translate-y-1/2"></div>
            <div className="absolute top-0 left-1/2 w-1 h-full bg-green-400 transform -translate-x-1/2"></div>
            <div 
              className="absolute top-0 left-0 w-full h-full border border-blue-500"
            ></div>
          </div>
        </div>
        <div className="text-center mt-4 italic">Just as 3D space needs 3 coordinates, 64D space needs 64 coordinates</div>
      </div>
    </div>
  );
};

const ComputerMemoryRepresentation: React.FC<StepProps> = ({ step }) => {
  return (
    <div className="flex flex-col items-center justify-center p-4">
      <h2 className="text-2xl font-bold mb-8">Computer Memory Representation</h2>
      <div className="w-full max-w-2xl bg-green-50 rounded-lg p-6 shadow-md">
        <div className="text-lg mb-4">In memory, a 64-dimensional vector is stored as a contiguous array:</div>
        <div className="font-mono text-sm bg-white p-3 rounded overflow-x-auto">
          <span className="text-green-600">float vector[64] = {'{'}</span>
          <span className="text-purple-500">0.24, -1.42, 0.35, 0.91, 0.02, -0.56, 0.78, 1.21, -0.63, 0.44, ...</span>
          <span className="text-green-600">{'}'};</span>
        </div>
        
        <div className="mt-8 text-lg">Visual representation of memory allocation:</div>
        <div className="mt-4 flex flex-wrap justify-center gap-1">
          {Array.from({ length: 64 }).map((_, i) => (
            <div 
              key={i} 
              className="w-8 h-8 bg-green-200 border border-green-400 flex items-center justify-center text-xs"
              style={{ 
                animation: `pulse 2s infinite ${i * 0.05}s`,
                backgroundColor: i % 4 === 0 ? 'rgb(187, 247, 208)' : 'rgb(167, 243, 208)'
              }}
            >
              {i}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

const NeuralNetworkRepresentation: React.FC<StepProps> = ({ step }) => {
  return (
    <div className="flex flex-col items-center justify-center p-4">
      <h2 className="text-2xl font-bold mb-8">Neural Network Representation</h2>
      <div className="w-full max-w-2xl bg-purple-50 rounded-lg p-6 shadow-md">
        <div className="text-lg mb-4">In a neural network layer with 64 channels/neurons:</div>
        <div className="flex justify-center">
          <div className="relative w-full max-w-lg h-64">
            {/* Input layer */}
            <div className="absolute left-0 top-1/2 transform -translate-y-1/2 flex flex-col gap-2">
              {Array.from({ length: 5 }).map((_, i) => (
                <div key={`input-${i}`} className="w-4 h-4 bg-purple-300 rounded-full"></div>
              ))}
            </div>
            
            {/* Hidden layer with 64 neurons */}
            <div className="absolute left-1/2 top-1/2 transform -translate-x-1/2 -translate-y-1/2 grid grid-cols-8 gap-1">
              {Array.from({ length: 64 }).map((_, i) => (
                <div 
                  key={`hidden-${i}`} 
                  className="w-4 h-4 bg-purple-400 rounded-full"
                  style={{ 
                    animation: `pulse 2s infinite ${i * 0.03}s`,
                    opacity: 0.5 + (Math.sin(i * 0.2) * 0.5)
                  }}
                ></div>
              ))}
            </div>
            
            {/* Output layer */}
            <div className="absolute right-0 top-1/2 transform -translate-y-1/2 flex flex-col gap-2">
              {Array.from({ length: 3 }).map((_, i) => (
                <div key={`output-${i}`} className="w-4 h-4 bg-purple-300 rounded-full"></div>
              ))}
            </div>
            
            {/* Connections */}
            <svg className="absolute top-0 left-0 w-full h-full" style={{ zIndex: -1 }}>
              {Array.from({ length: 20 }).map((_, i) => (
                <line 
                  key={`line-${i}`} 
                  x1="10%" 
                  y1={`${20 + i * 5}%`} 
                  x2="90%" 
                  y2={`${30 + i * 3}%`} 
                  stroke="rgba(192, 132, 252, 0.3)" 
                  strokeWidth="1"
                />
              ))}
            </svg>
          </div>
        </div>
        
        <div className="text-center mt-4">
          <p className="text-lg">Each neuron in the hidden layer represents one dimension of the 64D space</p>
          <p className="text-sm text-purple-600 mt-2">The activation values form a 64-dimensional vector</p>
        </div>
      </div>
    </div>
  );
};

const VisualizingHigherDimensions: React.FC<StepProps> = ({ step }) => {
  return (
    <div className="flex flex-col items-center justify-center p-4">
      <h2 className="text-2xl font-bold mb-8">Visualizing Higher Dimensions</h2>
      <div className="w-full max-w-2xl bg-yellow-50 rounded-lg p-6 shadow-md">
        <div className="text-lg mb-4">We can use dimensionality reduction to project 64D space to 2D:</div>
        <div className="flex justify-center">
          <div className="relative w-full max-w-lg h-64">
            {/* 2D projection visualization */}
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-48 h-48 bg-yellow-100 rounded-full relative overflow-hidden">
                {/* Points representing 64D data projected to 2D */}
                {Array.from({ length: 100 }).map((_, i) => (
                  <div 
                    key={`point-${i}`}
                    className="absolute w-2 h-2 bg-yellow-500 rounded-full"
                    style={{ 
                      left: `${30 + Math.sin(i * 0.2) * 40}%`,
                      top: `${30 + Math.cos(i * 0.3) * 40}%`,
                      animation: `float 3s infinite ${i * 0.05}s`,
                      opacity: 0.7
                    }}
                  ></div>
                ))}
                
                {/* Clusters */}
                <div className="absolute top-1/4 left-1/4 w-1/2 h-1/2 bg-yellow-200 rounded-full opacity-30"></div>
                <div className="absolute top-1/3 left-1/3 w-1/3 h-1/3 bg-yellow-300 rounded-full opacity-30"></div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="text-center mt-4">
          <p className="text-lg">Dimensionality reduction techniques like t-SNE or UMAP</p>
          <p className="text-sm text-yellow-600 mt-2">Preserve the structure and relationships of the high-dimensional data</p>
        </div>
      </div>
    </div>
  );
};

const FeatureVisualization: React.FC<StepProps> = ({ step }) => {
  return (
    <div className="flex flex-col items-center justify-center p-4">
      <h2 className="text-2xl font-bold mb-8">Feature Visualization</h2>
      <div className="w-full max-w-2xl bg-red-50 rounded-lg p-6 shadow-md">
        <div className="text-lg mb-4">We can visualize what each dimension "responds to" in CNN feature maps:</div>
        <div className="flex justify-center">
          <div className="relative w-full max-w-lg h-64">
            {/* Feature map visualization */}
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="grid grid-cols-8 gap-1">
                {Array.from({ length: 64 }).map((_, i) => (
                  <div 
                    key={`feature-${i}`}
                    className="w-8 h-8 bg-red-200 rounded"
                    style={{ 
                      animation: `pulse 2s infinite ${i * 0.03}s`,
                      backgroundImage: `radial-gradient(circle, rgba(239, 68, 68, ${0.3 + Math.sin(i * 0.2) * 0.3}) 0%, rgba(239, 68, 68, 0.1) 70%)`,
                      backgroundSize: 'cover'
                    }}
                  ></div>
                ))}
              </div>
            </div>
          </div>
        </div>
        
        <div className="text-center mt-4">
          <p className="text-lg">Each dimension can represent a specific feature or pattern</p>
          <p className="text-sm text-red-600 mt-2">In CNNs, these might be edges, textures, or shapes</p>
        </div>
      </div>
    </div>
  );
};

const PracticalCNNExample: React.FC<StepProps> = ({ step }) => {
  return (
    <div className="flex flex-col items-center justify-center p-4">
      <h2 className="text-2xl font-bold mb-8">Practical CNN Example</h2>
      <div className="w-full max-w-2xl bg-indigo-50 rounded-lg p-6 shadow-md">
        <div className="text-lg mb-4">In a U-Net or similar CNN architecture:</div>
        <div className="flex justify-center">
          <div className="relative w-full max-w-lg h-64">
            {/* CNN architecture visualization */}
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-full h-full flex flex-col justify-between">
                {/* Input */}
                <div className="w-full h-8 bg-indigo-200 rounded"></div>
                
                {/* Encoder path */}
                <div className="w-full flex justify-between">
                  <div className="w-1/4 h-16 bg-indigo-300 rounded"></div>
                  <div className="w-1/4 h-24 bg-indigo-400 rounded"></div>
                  <div className="w-1/4 h-32 bg-indigo-500 rounded"></div>
                  <div className="w-1/4 h-24 bg-indigo-400 rounded"></div>
                </div>
                
                {/* Bottleneck (64D representation) */}
                <div className="w-full h-16 bg-indigo-600 rounded flex items-center justify-center">
                  <div className="text-white text-sm font-bold">64D Space</div>
                </div>
                
                {/* Decoder path */}
                <div className="w-full flex justify-between">
                  <div className="w-1/4 h-24 bg-indigo-400 rounded"></div>
                  <div className="w-1/4 h-32 bg-indigo-500 rounded"></div>
                  <div className="w-1/4 h-24 bg-indigo-400 rounded"></div>
                  <div className="w-1/4 h-16 bg-indigo-300 rounded"></div>
                </div>
                
                {/* Output */}
                <div className="w-full h-8 bg-indigo-200 rounded"></div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="text-center mt-4">
          <p className="text-lg">The bottleneck layer compresses the input into a 64-dimensional representation</p>
          <p className="text-sm text-indigo-600 mt-2">This compact representation captures the essential features</p>
        </div>
      </div>
    </div>
  );
};

const AnimationControls: React.FC<AnimationControlsProps> = ({
  step,
  totalSteps,
  autoplay,
  onStepChange,
  onAutoplayToggle
}) => {
  const handlePrevious = () => {
    onStepChange((step - 1 + totalSteps) % totalSteps);
  };

  const handleNext = () => {
    onStepChange((step + 1) % totalSteps);
  };

  return (
    <div className="p-4 bg-gray-100 flex justify-between items-center">
      <button 
        onClick={handlePrevious}
        className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50"
        aria-label="Previous step"
      >
        Previous
      </button>
      
      <div className="flex items-center">
        <span className="mr-4" aria-live="polite">Step {step + 1} of {totalSteps}</span>
        <button 
          onClick={onAutoplayToggle}
          className={`px-4 py-2 rounded focus:outline-none focus:ring-2 focus:ring-opacity-50 ${
            autoplay ? 'bg-red-500 text-white focus:ring-red-500' : 'bg-green-500 text-white focus:ring-green-500'
          }`}
          aria-label={autoplay ? 'Pause animation' : 'Start animation'}
        >
          {autoplay ? 'Pause' : 'Autoplay'}
        </button>
      </div>
      
      <button 
        onClick={handleNext}
        className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50"
        aria-label="Next step"
      >
        Next
      </button>
    </div>
  );
};

const DimensionalSpaceAnimation: React.FC = () => {
  const [step, setStep] = useState<number>(0);
  const [autoplay, setAutoplay] = useState<boolean>(false);
  const totalSteps = 6;
  const ANIMATION_INTERVAL = 3500;
  
  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (autoplay) {
      timer = setInterval(() => {
        setStep((prevStep) => (prevStep + 1) % totalSteps);
      }, ANIMATION_INTERVAL);
    }
    return () => clearInterval(timer);
  }, [autoplay]);

  const renderStep = () => {
    switch(step) {
      case 0:
        return <MathematicalRepresentation step={step} />;
      case 1:
        return <ComputerMemoryRepresentation step={step} />;
      case 2:
        return <NeuralNetworkRepresentation step={step} />;
      case 3:
        return <VisualizingHigherDimensions step={step} />;
      case 4:
        return <FeatureVisualization step={step} />;
      case 5:
        return <PracticalCNNExample step={step} />;
      default:
        return null;
    }
  };

  const handleAutoplayToggle = () => {
    setAutoplay(!autoplay);
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <style>{globalStyles}</style>
      <header className="p-4 bg-gray-800 text-white">
        <h1 className="text-2xl font-bold text-center">Understanding 64-Dimensional Space in Neural Networks</h1>
      </header>
      
      <main className="flex-grow flex flex-col items-center justify-center p-4">
        <div className="transition-opacity duration-500 ease-in-out">
          {renderStep()}
        </div>
      </main>
      
      <AnimationControls
        step={step}
        totalSteps={totalSteps}
        autoplay={autoplay}
        onStepChange={setStep}
        onAutoplayToggle={handleAutoplayToggle}
      />
    </div>
  );
};

export default DimensionalSpaceAnimation; 