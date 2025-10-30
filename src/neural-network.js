///////////
// Types //
///////////

/*******************
 * GraphicsContext *
 *******************
 *
 * Creates a graphics context record that holds WebGL state and compiled shader programs.
 * @returns {{
 *   gl: WebGL2RenderingContext | null,
 *   programs: {
 *     forwardPropagation: WebGLProgram | null,
 *     backwardPropagation: WebGLProgram | null,
 *     weightUpdate: WebGLProgram | null,
 *     gradientComputation: WebGLProgram | null
 *   },
 *   quadBuffer: WebGLBuffer | null
 * }} Graphics context record
 */
function createGraphicsContext() {
    return {
        gl: null,
        programs: {
            forwardPropagation: null,
            backwardPropagation: null,
            weightUpdate: null,
            gradientComputation: null
        },
        quadBuffer: null
    };
}

/**********************
 * LayerSpecification *
 **********************
 *
 * Creates a layer specification record with dimensions and activation function.
 * @param {number} inputDimension - Number of inputs to this layer
 * @param {number} outputDimension - Number of outputs from this layer
 * @param {string} activationFunction - Activation function name ('sigmoid', 'relu', 'tanh', 'linear')
 * @returns {{
 *   inputDimension: number,
 *   outputDimension: number,
 *   activationFunction: string,
 *   activationCode: number
 * }} Layer specification record
 */
function createLayerSpecification(inputDimension, outputDimension, activationFunction) {
    console.assert(typeof inputDimension === 'number' && inputDimension > 0, 'inputDimension must be positive number');
    console.assert(typeof outputDimension === 'number' && outputDimension > 0, 'outputDimension must be positive number');
    console.assert(typeof activationFunction === 'string', 'activationFunction must be string');

    return {
        inputDimension,
        outputDimension,
        activationFunction,
        activationCode: getActivationCode(activationFunction)
    };
}

/*******************
 * LayerParameters *
 *******************
 *
 * Creates a layer parameters record with uninitialized weight and bias arrays.
 * @param {number} inputDimension - Number of inputs
 * @param {number} outputDimension - Number of outputs
 * @returns {{
 *   weights: Float32Array,
 *   biases: Float32Array
 * }} Layer parameters record
 */
function createLayerParameters(inputDimension, outputDimension) {
    console.assert(typeof inputDimension === 'number' && inputDimension > 0, 'inputDimension must be positive number');
    console.assert(typeof outputDimension === 'number' && outputDimension > 0, 'outputDimension must be positive number');

    return {
        weights: new Float32Array(inputDimension * outputDimension),
        biases: new Float32Array(outputDimension)
    };
}

/******************
 * LayerResources *
 ******************
 *
 * Creates a layer resources record with GPU textures and framebuffers.
 * All fields are initially null and populated during initialization.
 * @returns {{
 *   weightsPrimary: WebGLTexture | null,
 *   weightsSecondary: WebGLTexture | null,
 *   weightsActive: WebGLTexture | null,
 *   biasesPrimary: WebGLTexture | null,
 *   biasesSecondary: WebGLTexture | null,
 *   biasesActive: WebGLTexture | null,
 *   outputs: WebGLTexture | null,
 *   deltasPrimary: WebGLTexture | null,
 *   deltasSecondary: WebGLTexture | null,
 *   deltasActive: WebGLTexture | null,
 *   weightGradients: WebGLTexture | null,
 *   framebuffers: Object
 * }} Layer resources record
 */
function createLayerResources() {
    return {
        weightsPrimary: null,
        weightsSecondary: null,
        weightsActive: null,
        biasesPrimary: null,
        biasesSecondary: null,
        biasesActive: null,
        outputs: null,
        deltasPrimary: null,
        deltasSecondary: null,
        deltasActive: null,
        weightGradients: null,
        framebuffers: {}
    };
}

/****************
 * NetworkState *
 ****************
 *
 * Creates a network state record that holds all layers and GPU resources.
 * @returns {{
 *   layers: Array<Object>,
 *   parameters: Array<Object>,
 *   resources: Array<Object>,
 *   inputTexture: WebGLTexture | null,
 *   targetTexture: WebGLTexture | null,
 *   inputDimension: number,
 *   targetDimension: number
 * }} Network state record
 */
function createNetworkState() {
    return {
        layers: [],
        parameters: [],
        resources: [],
        inputTexture: null,
        targetTexture: null,
        inputDimension: 0,
        targetDimension: 0
    };
}

//////////////////////////
// Activation Functions //
//////////////////////////

/*********************
 * getActivationCode *
 *********************
 *
 * Maps activation function name to integer code for GPU shader.
 * @param {string} name - Activation function name
 * @returns {number} Activation code (0=linear, 1=sigmoid, 2=relu, 3=tanh)
 */
function getActivationCode(name) {
    console.assert(typeof name === 'string', 'name must be string');

    const codes = {
        'linear': 0,
        'sigmoid': 1,
        'relu': 2,
        'tanh': 3
    };
    return codes[name] || 0;
}

/****************
 * applySigmoid *
 ****************
 *
 * Applies sigmoid activation function: 1 / (1 + e^-x)
 * @param {number} x - Input value
 * @returns {number} Output in range (0, 1)
 */
function applySigmoid(x) {
    console.assert(typeof x === 'number', 'x must be number');
    return 1.0 / (1.0 + Math.exp(-x));
}

/*************
 * applyRelu *
 *************
 *
 * Applies ReLU activation function: max(0, x)
 * @param {number} x - Input value
 * @returns {number} Output, zero if negative
 */
function applyRelu(x) {
    console.assert(typeof x === 'number', 'x must be number');
    return Math.max(0.0, x);
}

/*************
 * applyTanh *
 *************
 *
 * Applies hyperbolic tangent activation function.
 * @param {number} x - Input value
 * @returns {number} Output in range (-1, 1)
 */
function applyTanh(x) {
    console.assert(typeof x === 'number', 'x must be number');
    return Math.tanh(x);
}

/***************
 * applyLinear *
 ***************
 *
 * Applies linear activation function (identity).
 * @param {number} x - Input value
 * @returns {number} Output equal to input
 */
function applyLinear(x) {
    console.assert(typeof x === 'number', 'x must be number');
    return x;
}

//////////////
// Graphics //
//////////////

/*****************************
 * initializeGraphicsContext *
 *****************************
 *
 * Initializes WebGL2 context with required extensions and shader programs.
 * @returns {Object} Initialized graphics context
 * @throws {Error} If WebGL2 is not supported
 */
function initializeGraphicsContext(canvasOrOptions = null) {
    const context = createGraphicsContext();

    let canvas;

    // Check if running in Node.js
    if (typeof window === 'undefined') {
        // Node.js environment
        if (!canvasOrOptions) {
            throw new Error('In Node.js environment, you must provide a WebGL context or canvas');
        }
        // Allow passing a context directly or an object with getContext
        if (typeof canvasOrOptions.getContext === 'function') {
            canvas = canvasOrOptions;
        } else {
            // Assume it's already a WebGL context
            context.gl = canvasOrOptions;
        }
    } else {
        // Browser environment
        if (canvasOrOptions instanceof HTMLCanvasElement) {
            canvas = canvasOrOptions;
        } else {
            canvas = document.createElement('canvas');
        }
    }

    // Get WebGL context if we have a canvas
    if (canvas && !context.gl) {
        context.gl = canvas.getContext('webgl2');
    }

    if (!context.gl) {
        throw new Error('WebGL2 not supported');
    }

    context.gl.getExtension('EXT_color_buffer_float');
    context.gl.getExtension('OES_texture_float_linear');

    createShaderPrograms(context);
    createQuadBuffer(context);

    console.assert(context.gl !== null, 'WebGL context must be initialized');
    console.assert(context.programs.forwardPropagation !== null, 'Forward program must be compiled');
    console.assert(context.programs.backwardPropagation !== null, 'Backward program must be compiled');
    console.assert(context.quadBuffer !== null, 'Quad buffer must be created');

    return context;
}

/********************
 * createQuadBuffer *
 ********************
 *
 * Creates a fullscreen quad buffer for shader rendering.
 * The quad covers the entire viewport from (-1,-1) to (1,1).
 * @param {Object} context - Graphics context
 */
function createQuadBuffer(context) {
    console.assert(context.gl !== null, 'WebGL context must exist');

    const gl = context.gl;
    context.quadBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, context.quadBuffer);
    gl.bufferData(
        gl.ARRAY_BUFFER,
        new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]),
        gl.STATIC_DRAW
    );

    console.assert(context.quadBuffer !== null, 'Quad buffer creation failed');
}

/*****************
 * compileShader *
 *****************
 *
 * Compiles a shader from source code.
 * @param {Object} context - Graphics context
 * @param {number} type - Shader type (gl.VERTEX_SHADER or gl.FRAGMENT_SHADER)
 * @param {string} source - GLSL shader source code
 * @returns {WebGLShader} Compiled shader
 * @throws {Error} If shader compilation fails
 */
function compileShader(context, type, source) {
    console.assert(context.gl !== null, 'WebGL context must exist');
    console.assert(typeof source === 'string' && source.length > 0, 'source must be non-empty string');

    const gl = context.gl;
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        throw new Error('Shader compile error: ' + gl.getShaderInfoLog(shader));
    }

    console.assert(shader !== null, 'Shader compilation failed');
    return shader;
}

/***************
 * linkProgram *
 ***************
 *
 * Links vertex and fragment shaders into a program.
 * @param {Object} context - Graphics context
 * @param {WebGLShader} vertexShader - Compiled vertex shader
 * @param {WebGLShader} fragmentShader - Compiled fragment shader
 * @returns {WebGLProgram} Linked shader program
 * @throws {Error} If program linking fails
 */
function linkProgram(context, vertexShader, fragmentShader) {
    console.assert(context.gl !== null, 'WebGL context must exist');
    console.assert(vertexShader !== null, 'vertexShader must exist');
    console.assert(fragmentShader !== null, 'fragmentShader must exist');

    const gl = context.gl;
    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        throw new Error('Program link error: ' + gl.getProgramInfoLog(program));
    }

    console.assert(program !== null, 'Program linking failed');
    return program;
}

/************************
 * createShaderPrograms *
 ************************
 *
 * Compiles and links all shader programs used by the neural network.
 * Creates programs for forward propagation, backpropagation, gradient computation, and weight updates.
 * @param {Object} context - Graphics context
 */
function createShaderPrograms(context) {
    console.assert(context.gl !== null, 'WebGL context must exist');

    const gl = context.gl;

    const vertexShaderSource = `#version 300 es
        precision highp float;
        in vec2 position;
        out vec2 textureCoordinate;
        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
            textureCoordinate = position * 0.5 + 0.5;
        }
    `;

    const forwardFragmentSource = `#version 300 es
        precision highp float;
        uniform sampler2D weights;
        uniform sampler2D inputValues;
        uniform sampler2D biases;
        uniform int inputSize;
        uniform int outputSize;
        uniform int activation;
        in vec2 textureCoordinate;
        out vec4 fragmentColor;

        float sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }
        float relu(float x) { return max(0.0, x); }
        float tanhActivation(float x) { return tanh(x); }

        float applyActivation(float x, int activationType) {
            if (activationType == 1) return sigmoid(x);
            if (activationType == 2) return relu(x);
            if (activationType == 3) return tanhActivation(x);
            return x;
        }

        void main() {
            int neuronIndex = int(textureCoordinate.y * float(outputSize));
            if (neuronIndex >= outputSize) {
                fragmentColor = vec4(0.0);
                return;
            }

            float sum = texture(biases, vec2(float(neuronIndex) / float(outputSize), 0.5)).r;

            for (int i = 0; i < 1024; i++) {
                if (i >= inputSize) break;
                float weight = texture(weights, vec2(float(i) / float(inputSize), float(neuronIndex) / float(outputSize))).r;
                float inputValue = texture(inputValues, vec2(float(i) / float(inputSize), 0.5)).r;
                sum += weight * inputValue;
            }

            float outputValue = applyActivation(sum, activation);
            fragmentColor = vec4(outputValue, sum, 0.0, 1.0);
        }
    `;

    const backwardFragmentSource = `#version 300 es
        precision highp float;
        uniform sampler2D previousOutputs;
        uniform sampler2D rawSums;
        uniform sampler2D targets;
        uniform sampler2D nextLayerWeights;
        uniform sampler2D nextLayerDeltas;
        uniform int currentSize;
        uniform int nextSize;
        uniform int activation;
        uniform bool isOutputLayer;
        in vec2 textureCoordinate;
        out vec4 fragmentColor;

        float sigmoidDerivative(float x) {
            float s = 1.0 / (1.0 + exp(-x));
            return s * (1.0 - s);
        }

        float reluDerivative(float x) { return x > 0.0 ? 1.0 : 0.0; }
        float tanhDerivative(float x) { return 1.0 - tanh(x) * tanh(x); }

        float getDerivative(float x, int activationType) {
            if (activationType == 1) return sigmoidDerivative(x);
            if (activationType == 2) return reluDerivative(x);
            if (activationType == 3) return tanhDerivative(x);
            return 1.0;
        }

        void main() {
            int neuronIndex = int(textureCoordinate.y * float(currentSize));
            if (neuronIndex >= currentSize) {
                fragmentColor = vec4(0.0);
                return;
            }

            float delta;
            float rawSum = texture(rawSums, vec2(float(neuronIndex) / float(currentSize), 0.5)).g;
            float derivative = getDerivative(rawSum, activation);

            if (isOutputLayer) {
                float outputValue = texture(previousOutputs, vec2(float(neuronIndex) / float(currentSize), 0.5)).r;
                float targetValue = texture(targets, vec2(float(neuronIndex) / float(currentSize), 0.5)).r;
                delta = (outputValue - targetValue) * derivative;
            } else {
                float errorSum = 0.0;
                for (int i = 0; i < 1024; i++) {
                    if (i >= nextSize) break;
                    float weight = texture(nextLayerWeights, vec2(float(neuronIndex) / float(currentSize), float(i) / float(nextSize))).r;
                    float nextDelta = texture(nextLayerDeltas, vec2(float(i) / float(nextSize), 0.5)).r;
                    errorSum += weight * nextDelta;
                }
                delta = errorSum * derivative;
            }

            fragmentColor = vec4(delta, 0.0, 0.0, 1.0);
        }
    `;

    const gradientFragmentSource = `#version 300 es
        precision highp float;
        uniform sampler2D deltas;
        uniform sampler2D previousOutputs;
        uniform int outputSize;
        uniform int inputSize;
        in vec2 textureCoordinate;
        out vec4 fragmentColor;

        void main() {
            int inputIndex = int(textureCoordinate.x * float(inputSize));
            int outputIndex = int(textureCoordinate.y * float(outputSize));

            if (inputIndex >= inputSize || outputIndex >= outputSize) {
                fragmentColor = vec4(0.0);
                return;
            }

            float delta = texture(deltas, vec2(float(outputIndex) / float(outputSize), 0.5)).r;
            float previousOutput = texture(previousOutputs, vec2(float(inputIndex) / float(inputSize), 0.5)).r;

            float gradient = delta * previousOutput;
            fragmentColor = vec4(gradient, 0.0, 0.0, 1.0);
        }
    `;

    const updateWeightsSource = `#version 300 es
        precision highp float;
        uniform sampler2D weights;
        uniform sampler2D gradients;
        uniform float learningRate;
        in vec2 textureCoordinate;
        out vec4 fragmentColor;

        void main() {
            float weight = texture(weights, textureCoordinate).r;
            float gradient = texture(gradients, textureCoordinate).r;
            float newWeight = weight - learningRate * gradient;
            fragmentColor = vec4(newWeight, 0.0, 0.0, 1.0);
        }
    `;

    const vertexShader = compileShader(context, gl.VERTEX_SHADER, vertexShaderSource);

    const forwardFragment = compileShader(context, gl.FRAGMENT_SHADER, forwardFragmentSource);
    const backwardFragment = compileShader(context, gl.FRAGMENT_SHADER, backwardFragmentSource);
    const gradientFragment = compileShader(context, gl.FRAGMENT_SHADER, gradientFragmentSource);
    const updateFragment = compileShader(context, gl.FRAGMENT_SHADER, updateWeightsSource);

    context.programs.forwardPropagation = linkProgram(context, vertexShader, forwardFragment);
    context.programs.backwardPropagation = linkProgram(context, vertexShader, backwardFragment);
    context.programs.gradientComputation = linkProgram(context, vertexShader, gradientFragment);
    context.programs.weightUpdate = linkProgram(context, vertexShader, updateFragment);

    console.assert(context.programs.forwardPropagation !== null, 'Forward program must be linked');
    console.assert(context.programs.backwardPropagation !== null, 'Backward program must be linked');
    console.assert(context.programs.gradientComputation !== null, 'Gradient program must be linked');
    console.assert(context.programs.weightUpdate !== null, 'Weight update program must be linked');
}

/*****************
 * createTexture *
 *****************
 *
 * Creates a single-channel floating-point texture for storing neural network data.
 * @param {Object} context - Graphics context
 * @param {number} width - Texture width in pixels
 * @param {number} height - Texture height in pixels
 * @param {Float32Array|null} data - Optional initial data
 * @returns {WebGLTexture} Created texture
 */
function createTexture(context, width, height, data = null) {
    console.assert(context.gl !== null, 'WebGL context must exist');
    console.assert(typeof width === 'number' && width > 0, 'width must be positive number');
    console.assert(typeof height === 'number' && height > 0, 'height must be positive number');
    console.assert(data === null || data instanceof Float32Array, 'data must be null or Float32Array');

    const gl = context.gl;
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, data);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    console.assert(texture !== null, 'Texture creation failed');
    return texture;
}

/*********************
 * createFramebuffer *
 *********************
 *
 * Creates a framebuffer for rendering to a texture.
 * @param {Object} context - Graphics context
 * @param {WebGLTexture} texture - Texture to attach to framebuffer
 * @returns {WebGLFramebuffer} Created framebuffer
 */
function createFramebuffer(context, texture) {
    console.assert(context.gl !== null, 'WebGL context must exist');
    console.assert(texture !== null, 'texture must exist');

    const gl = context.gl;
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);

    console.assert(framebuffer !== null, 'Framebuffer creation failed');
    return framebuffer;
}

/*********************
 * uploadTextureData *
 *********************
 *
 * Uploads data to an existing texture.
 * @param {Object} context - Graphics context
 * @param {WebGLTexture} texture - Target texture
 * @param {Array|Float32Array} data - Data to upload
 * @param {number} width - Data width
 * @param {number} height - Data height
 */
function uploadTextureData(context, texture, data, width, height) {
    console.assert(context.gl !== null, 'WebGL context must exist');
    console.assert(texture !== null, 'texture must exist');
    console.assert(data !== null && data.length > 0, 'data must be non-empty');
    console.assert(typeof width === 'number' && width > 0, 'width must be positive number');
    console.assert(typeof height === 'number' && height > 0, 'height must be positive number');

    const gl = context.gl;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, height, gl.RED, gl.FLOAT, new Float32Array(data));
}

/*******************
 * readTextureData *
 *******************
 *
 * Reads data from the currently bound framebuffer.
 * Returns only the red channel values from the RGBA output.
 * @param {Object} context - Graphics context
 * @param {WebGLTexture} texture - Texture to read (currently unused, reads from bound framebuffer)
 * @param {number} width - Data width
 * @param {number} height - Data height
 * @returns {Float32Array} Extracted data
 */
function readTextureData(context, texture, width, height) {
    console.assert(context.gl !== null, 'WebGL context must exist');
    console.assert(typeof width === 'number' && width > 0, 'width must be positive number');
    console.assert(typeof height === 'number' && height > 0, 'height must be positive number');

    const gl = context.gl;
    const buffer = new Float32Array(width * height * 4);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.FLOAT, buffer);

    const result = new Float32Array(width * height);
    for (let i = 0; i < width * height; i++) {
        result[i] = buffer[i * 4];
    }
    return result;
}

/**************
 * renderQuad *
 **************
 *
 * Renders a fullscreen quad using the currently bound shader program.
 * @param {Object} context - Graphics context
 */
function renderQuad(context) {
    console.assert(context.gl !== null, 'WebGL context must exist');
    console.assert(context.quadBuffer !== null, 'Quad buffer must exist');

    const gl = context.gl;
    const currentProgram = gl.getParameter(gl.CURRENT_PROGRAM);
    if (!currentProgram) return;

    gl.bindBuffer(gl.ARRAY_BUFFER, context.quadBuffer);
    const positionLocation = gl.getAttribLocation(currentProgram, 'position');

    if (positionLocation >= 0) {
        gl.enableVertexAttribArray(positionLocation);
        gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);
    }

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

////////////////////
// Initialization //
////////////////////

/*****************************
 * initializeLayerParameters *
 *****************************
 *
 * Initializes layer parameters using Xavier initialization.
 * Xavier initialization generates properly scaled random weights
 * to prevent vanishing or exploding gradients during training.
 * Scale factor is sqrt(6 / (input_dimension + output_dimension))
 * @param {Object} specification - Layer specification
 * @returns {Object} Initialized layer parameters with weights and biases
 */
function initializeLayerParameters(specification) {
    console.assert(specification !== null, 'specification must exist');
    console.assert(specification.inputDimension > 0, 'inputDimension must be positive');
    console.assert(specification.outputDimension > 0, 'outputDimension must be positive');

    const parameters = createLayerParameters(
        specification.inputDimension,
        specification.outputDimension
    );

    const limit = Math.sqrt(6.0 / (specification.inputDimension + specification.outputDimension));

    for (let i = 0; i < parameters.weights.length; i++) {
        parameters.weights[i] = (Math.random() * 2.0 - 1.0) * limit;
    }

    for (let i = 0; i < parameters.biases.length; i++) {
        parameters.biases[i] = (Math.random() * 2.0 - 1.0) * limit;
    }

    return parameters;
}

/****************************
 * initializeLayerResources *
 ****************************
 *
 * Creates GPU resources (textures and framebuffers) for a layer.
 * Uses double-buffering for weights, biases, and deltas to enable ping-pong updates.
 * @param {Object} context - Graphics context
 * @param {Object} specification - Layer specification
 * @param {Object} parameters - Initialized layer parameters
 * @returns {Object} Layer resources with all GPU textures and framebuffers
 */
function initializeLayerResources(context, specification, parameters) {
    console.assert(context.gl !== null, 'WebGL context must exist');
    console.assert(specification !== null, 'specification must exist');
    console.assert(parameters !== null, 'parameters must exist');
    console.assert(parameters.weights.length === specification.inputDimension * specification.outputDimension,
        'weights array must match layer dimensions');
    console.assert(parameters.biases.length === specification.outputDimension,
        'biases array must match output dimension');

    const resources = createLayerResources();

    resources.weightsPrimary = createTexture(
        context,
        specification.inputDimension,
        specification.outputDimension,
        parameters.weights
    );
    resources.weightsSecondary = createTexture(
        context,
        specification.inputDimension,
        specification.outputDimension
    );
    resources.weightsActive = resources.weightsPrimary;

    resources.biasesPrimary = createTexture(
        context,
        specification.outputDimension,
        1,
        parameters.biases
    );
    resources.biasesSecondary = createTexture(
        context,
        specification.outputDimension,
        1
    );
    resources.biasesActive = resources.biasesPrimary;

    resources.outputs = createTexture(context, specification.outputDimension, 1);
    resources.deltasPrimary = createTexture(context, specification.outputDimension, 1);
    resources.deltasSecondary = createTexture(context, specification.outputDimension, 1);
    resources.deltasActive = resources.deltasPrimary;
    resources.weightGradients = createTexture(
        context,
        specification.inputDimension,
        specification.outputDimension
    );

    resources.framebuffers.outputs = createFramebuffer(context, resources.outputs);
    resources.framebuffers.weightsPrimary = createFramebuffer(context, resources.weightsPrimary);
    resources.framebuffers.weightsSecondary = createFramebuffer(context, resources.weightsSecondary);
    resources.framebuffers.biasesPrimary = createFramebuffer(context, resources.biasesPrimary);
    resources.framebuffers.biasesSecondary = createFramebuffer(context, resources.biasesSecondary);
    resources.framebuffers.deltasPrimary = createFramebuffer(context, resources.deltasPrimary);
    resources.framebuffers.deltasSecondary = createFramebuffer(context, resources.deltasSecondary);
    resources.framebuffers.weightGradients = createFramebuffer(context, resources.weightGradients);

    console.assert(resources.weightsActive !== null, 'weights must be created');
    console.assert(resources.biasesActive !== null, 'biases must be created');
    console.assert(resources.outputs !== null, 'outputs must be created');

    return resources;
}

/*********************
 * addLayerToNetwork *
 *********************
 *
 * Adds a new layer to an existing network.
 * @param {Object} context - Graphics context
 * @param {Object} network - Network state
 * @param {number} inputDimension - Number of inputs
 * @param {number} outputDimension - Number of outputs
 * @param {string} activationFunction - Activation function name
 */
function addLayerToNetwork(context, network, inputDimension, outputDimension, activationFunction) {
    console.assert(context.gl !== null, 'WebGL context must exist');
    console.assert(network !== null, 'network must exist');
    console.assert(typeof inputDimension === 'number' && inputDimension > 0, 'inputDimension must be positive');
    console.assert(typeof outputDimension === 'number' && outputDimension > 0, 'outputDimension must be positive');

    const specification = createLayerSpecification(inputDimension, outputDimension, activationFunction);
    const parameters = initializeLayerParameters(specification);
    const resources = initializeLayerResources(context, specification, parameters);

    network.layers.push(specification);
    network.parameters.push(parameters);
    network.resources.push(resources);
}

/*********************
 * initializeNetwork *
 *********************
 *
 * Initializes a complete neural network from layer specifications.
 * @param {Object} context - Graphics context
 * @param {Array<Object>} layerSpecifications - Array of layer specifications
 * @returns {Object} Initialized network state
 */
function initializeNetwork(context, layerSpecifications) {
    console.assert(context.gl !== null, 'WebGL context must exist');
    console.assert(Array.isArray(layerSpecifications), 'layerSpecifications must be array');
    console.assert(layerSpecifications.length > 0, 'must have at least one layer');

    const network = createNetworkState();

    for (const spec of layerSpecifications) {
        addLayerToNetwork(
            context,
            network,
            spec.inputDimension,
            spec.outputDimension,
            spec.activationFunction
        );
    }

    console.assert(network.layers.length === layerSpecifications.length, 'all layers must be added');
    return network;
}

/////////////////////////
// Forward Propagation //
/////////////////////////

/*************************
 * computeNetworkForward *
 *************************
 *
 * Computes forward pass for a single layer using GPU.
 * Multiplies input by weights, adds biases, and applies activation function.
 * @param {Object} context - Graphics context
 * @param {Object} specification - Layer specification
 * @param {Object} resources - Layer GPU resources
 * @param {WebGLTexture} inputTexture - Input data texture
 * @param {number} inputDimension - Size of input vector
 * @returns {WebGLTexture} Output texture containing layer activations
 */
function computeLayerForward(context, specification, resources, inputTexture, inputDimension) {
    console.assert(context.gl !== null, 'WebGL context must exist');
    console.assert(specification !== null, 'specification must exist');
    console.assert(resources !== null, 'resources must exist');
    console.assert(inputTexture !== null, 'inputTexture must exist');
    console.assert(typeof inputDimension === 'number' && inputDimension > 0, 'inputDimension must be positive');

    const gl = context.gl;

    gl.useProgram(context.programs.forwardPropagation);
    gl.bindFramebuffer(gl.FRAMEBUFFER, resources.framebuffers.outputs);
    gl.viewport(0, 0, specification.outputDimension, 1);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, resources.weightsActive);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, inputTexture);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, resources.biasesActive);

    gl.uniform1i(gl.getUniformLocation(context.programs.forwardPropagation, 'weights'), 0);
    gl.uniform1i(gl.getUniformLocation(context.programs.forwardPropagation, 'inputValues'), 1);
    gl.uniform1i(gl.getUniformLocation(context.programs.forwardPropagation, 'biases'), 2);
    gl.uniform1i(gl.getUniformLocation(context.programs.forwardPropagation, 'inputSize'), inputDimension);
    gl.uniform1i(gl.getUniformLocation(context.programs.forwardPropagation, 'outputSize'), specification.outputDimension);
    gl.uniform1i(gl.getUniformLocation(context.programs.forwardPropagation, 'activation'), specification.activationCode);

    renderQuad(context);

    return resources.outputs;
}

/*************************
 * computeNetworkForward *
 *************************
 *
 * Computes forward pass through entire network.
 * @param {Object} context - Graphics context
 * @param {Object} network - Network state
 * @param {Array<number>|Float32Array} inputValues - Input vector
 * @returns {Float32Array} Network output
 */
function computeNetworkForward(context, network, inputValues) {
    console.assert(context.gl !== null, 'WebGL context must exist');
    console.assert(network !== null, 'network must exist');
    console.assert(network.layers.length > 0, 'network must have at least one layer');
    console.assert(inputValues !== null && inputValues.length > 0, 'inputValues must be non-empty');

    const gl = context.gl;

    if (!network.inputTexture || network.inputDimension !== inputValues.length) {
        network.inputDimension = inputValues.length;
        network.inputTexture = createTexture(context, inputValues.length, 1);
    }

    uploadTextureData(context, network.inputTexture, inputValues, inputValues.length, 1);

    let previousOutput = network.inputTexture;
    let previousDimension = inputValues.length;

    for (let i = 0; i < network.layers.length; i++) {
        previousOutput = computeLayerForward(
            context,
            network.layers[i],
            network.resources[i],
            previousOutput,
            previousDimension
        );
        previousDimension = network.layers[i].outputDimension;
    }

    const lastLayer = network.layers[network.layers.length - 1];
    gl.bindFramebuffer(gl.FRAMEBUFFER, network.resources[network.resources.length - 1].framebuffers.outputs);
    const result = readTextureData(context, previousOutput, lastLayer.outputDimension, 1);

    console.assert(result.length === lastLayer.outputDimension, 'output size must match last layer');
    return result;
}

/////////////////////
// Backpropagation //
/////////////////////

/**********************
 * computeLayerDeltas *
 **********************
 *
 * Computes error gradients (deltas) for a single layer using backpropagation.
 * For output layers: delta = (output - target) * activation_derivative
 * For hidden layers: delta = sum(next_weights * next_deltas) * activation_derivative
 * Uses double-buffering to swap between delta textures.
 * @param {Object} context - Graphics context
 * @param {number} layerIndex - Index of layer in network
 * @param {Object} network - Network state
 * @param {boolean} isOutputLayer - Whether this is the output layer
 */
function computeLayerDeltas(context, layerIndex, network, isOutputLayer) {
    console.assert(context.gl !== null, 'WebGL context must exist');
    console.assert(typeof layerIndex === 'number' && layerIndex >= 0, 'layerIndex must be non-negative');
    console.assert(layerIndex < network.layers.length, 'layerIndex must be within network bounds');
    console.assert(network !== null, 'network must exist');
    console.assert(typeof isOutputLayer === 'boolean', 'isOutputLayer must be boolean');

    const gl = context.gl;
    const layer = network.layers[layerIndex];
    const resources = network.resources[layerIndex];

    const currentDelta = resources.deltasActive;
    const targetDelta = (currentDelta === resources.deltasPrimary)
        ? resources.deltasSecondary
        : resources.deltasPrimary;

    const targetFramebuffer = (targetDelta === resources.deltasPrimary)
        ? resources.framebuffers.deltasPrimary
        : resources.framebuffers.deltasSecondary;

    gl.useProgram(context.programs.backwardPropagation);
    gl.bindFramebuffer(gl.FRAMEBUFFER, targetFramebuffer);
    gl.viewport(0, 0, layer.outputDimension, 1);

    gl.uniform1i(gl.getUniformLocation(context.programs.backwardPropagation, 'currentSize'), layer.outputDimension);
    gl.uniform1i(gl.getUniformLocation(context.programs.backwardPropagation, 'activation'), layer.activationCode);
    gl.uniform1i(gl.getUniformLocation(context.programs.backwardPropagation, 'isOutputLayer'), isOutputLayer);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, resources.outputs);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, resources.outputs);

    gl.uniform1i(gl.getUniformLocation(context.programs.backwardPropagation, 'previousOutputs'), 0);
    gl.uniform1i(gl.getUniformLocation(context.programs.backwardPropagation, 'rawSums'), 1);

    if (isOutputLayer) {
        console.assert(network.targetTexture !== null, 'target texture must exist for output layer');
        gl.activeTexture(gl.TEXTURE2);
        gl.bindTexture(gl.TEXTURE_2D, network.targetTexture);
        gl.uniform1i(gl.getUniformLocation(context.programs.backwardPropagation, 'targets'), 2);
    } else {
        console.assert(layerIndex + 1 < network.layers.length, 'next layer must exist for hidden layer');
        const nextLayer = network.resources[layerIndex + 1];
        gl.activeTexture(gl.TEXTURE2);
        gl.bindTexture(gl.TEXTURE_2D, nextLayer.weightsActive);
        gl.activeTexture(gl.TEXTURE3);
        gl.bindTexture(gl.TEXTURE_2D, nextLayer.deltasActive);
        gl.uniform1i(gl.getUniformLocation(context.programs.backwardPropagation, 'nextLayerWeights'), 2);
        gl.uniform1i(gl.getUniformLocation(context.programs.backwardPropagation, 'nextLayerDeltas'), 3);
        gl.uniform1i(gl.getUniformLocation(context.programs.backwardPropagation, 'nextSize'), network.layers[layerIndex + 1].outputDimension);
    }

    renderQuad(context);

    resources.deltasActive = targetDelta;
    gl.finish();
}

/**************************
 * computeWeightGradients *
 **************************
 *
 * Computes weight gradients for a single layer.
 * Gradient = delta * previous_layer_output
 * @param {Object} context - Graphics context
 * @param {number} layerIndex - Index of layer in network
 * @param {Object} network - Network state
 */
function computeWeightGradients(context, layerIndex, network) {
    console.assert(context.gl !== null, 'WebGL context must exist');
    console.assert(typeof layerIndex === 'number' && layerIndex >= 0, 'layerIndex must be non-negative');
    console.assert(layerIndex < network.layers.length, 'layerIndex must be within network bounds');
    console.assert(network !== null, 'network must exist');

    const gl = context.gl;
    const layer = network.layers[layerIndex];
    const resources = network.resources[layerIndex];

    gl.useProgram(context.programs.gradientComputation);
    gl.bindFramebuffer(gl.FRAMEBUFFER, resources.framebuffers.weightGradients);
    gl.viewport(0, 0, layer.inputDimension, layer.outputDimension);

    gl.uniform1i(gl.getUniformLocation(context.programs.gradientComputation, 'outputSize'), layer.outputDimension);
    gl.uniform1i(gl.getUniformLocation(context.programs.gradientComputation, 'inputSize'), layer.inputDimension);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, resources.deltasActive);
    gl.activeTexture(gl.TEXTURE1);

    if (layerIndex === 0) {
        console.assert(network.inputTexture !== null, 'input texture must exist for first layer');
        gl.bindTexture(gl.TEXTURE_2D, network.inputTexture);
    } else {
        gl.bindTexture(gl.TEXTURE_2D, network.resources[layerIndex - 1].outputs);
    }

    gl.uniform1i(gl.getUniformLocation(context.programs.gradientComputation, 'deltas'), 0);
    gl.uniform1i(gl.getUniformLocation(context.programs.gradientComputation, 'previousOutputs'), 1);

    renderQuad(context);
}

/**********************
 * updateLayerWeights *
 **********************
 *
 * Updates weights for a single layer using computed gradients.
 * new_weight = old_weight - learning_rate * gradient
 * Uses double-buffering to swap between weight textures.
 * @param {Object} context - Graphics context
 * @param {number} layerIndex - Index of layer in network
 * @param {Object} network - Network state
 * @param {number} learningRate - Learning rate for gradient descent
 */
function updateLayerWeights(context, layerIndex, network, learningRate) {
    console.assert(context.gl !== null, 'WebGL context must exist');
    console.assert(typeof layerIndex === 'number' && layerIndex >= 0, 'layerIndex must be non-negative');
    console.assert(layerIndex < network.layers.length, 'layerIndex must be within network bounds');
    console.assert(network !== null, 'network must exist');
    console.assert(typeof learningRate === 'number' && learningRate > 0, 'learningRate must be positive number');

    const gl = context.gl;
    const layer = network.layers[layerIndex];
    const resources = network.resources[layerIndex];

    const currentWeights = resources.weightsActive;
    const targetWeights = (currentWeights === resources.weightsPrimary)
        ? resources.weightsSecondary
        : resources.weightsPrimary;

    const targetFramebuffer = (targetWeights === resources.weightsPrimary)
        ? resources.framebuffers.weightsPrimary
        : resources.framebuffers.weightsSecondary;

    gl.useProgram(context.programs.weightUpdate);
    gl.bindFramebuffer(gl.FRAMEBUFFER, targetFramebuffer);
    gl.viewport(0, 0, layer.inputDimension, layer.outputDimension);

    gl.uniform1f(gl.getUniformLocation(context.programs.weightUpdate, 'learningRate'), learningRate);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, currentWeights);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, resources.weightGradients);

    gl.uniform1i(gl.getUniformLocation(context.programs.weightUpdate, 'weights'), 0);
    gl.uniform1i(gl.getUniformLocation(context.programs.weightUpdate, 'gradients'), 1);

    renderQuad(context);

    resources.weightsActive = targetWeights;
}

/*********************
 * updateLayerBiases *
 *********************
 *
 * Updates biases for a single layer using computed deltas.
 * new_bias = old_bias - learning_rate * delta
 * Uses double-buffering to swap between bias textures.
 * @param {Object} context - Graphics context
 * @param {number} layerIndex - Index of layer in network
 * @param {Object} network - Network state
 * @param {number} learningRate - Learning rate for gradient descent
 */
function updateLayerBiases(context, layerIndex, network, learningRate) {
    console.assert(context.gl !== null, 'WebGL context must exist');
    console.assert(typeof layerIndex === 'number' && layerIndex >= 0, 'layerIndex must be non-negative');
    console.assert(layerIndex < network.layers.length, 'layerIndex must be within network bounds');
    console.assert(network !== null, 'network must exist');
    console.assert(typeof learningRate === 'number' && learningRate > 0, 'learningRate must be positive number');

    const gl = context.gl;
    const layer = network.layers[layerIndex];
    const resources = network.resources[layerIndex];

    const currentBiases = resources.biasesActive;
    const targetBiases = (currentBiases === resources.biasesPrimary)
        ? resources.biasesSecondary
        : resources.biasesPrimary;

    const targetFramebuffer = (targetBiases === resources.biasesPrimary)
        ? resources.framebuffers.biasesPrimary
        : resources.framebuffers.biasesSecondary;

    gl.useProgram(context.programs.weightUpdate);
    gl.bindFramebuffer(gl.FRAMEBUFFER, targetFramebuffer);
    gl.viewport(0, 0, layer.outputDimension, 1);

    gl.uniform1f(gl.getUniformLocation(context.programs.weightUpdate, 'learningRate'), learningRate);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, currentBiases);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, resources.deltasActive);

    gl.uniform1i(gl.getUniformLocation(context.programs.weightUpdate, 'weights'), 0);
    gl.uniform1i(gl.getUniformLocation(context.programs.weightUpdate, 'gradients'), 1);

    renderQuad(context);

    resources.biasesActive = targetBiases;
}

/*********************************
 * computeNetworkBackpropagation *
 *********************************
 *
 * Performs backpropagation through the entire network.
 * Computes deltas for all layers (output to input) and gradients for all weights.
 * @param {Object} context - Graphics context
 * @param {Object} network - Network state
 */
function computeNetworkBackpropagation(context, network) {
    console.assert(context.gl !== null, 'WebGL context must exist');
    console.assert(network !== null, 'network must exist');
    console.assert(network.layers.length > 0, 'network must have at least one layer');

    for (let layerIndex = network.layers.length - 1; layerIndex >= 0; layerIndex--) {
        const isOutputLayer = layerIndex === network.layers.length - 1;
        computeLayerDeltas(context, layerIndex, network, isOutputLayer);
    }

    for (let layerIndex = 0; layerIndex < network.layers.length; layerIndex++) {
        computeWeightGradients(context, layerIndex, network);
    }
}

/***************************
 * updateNetworkParameters *
 ***************************
 *
 * Updates all network parameters (weights and biases) using computed gradients.
 * @param {Object} context - Graphics context
 * @param {Object} network - Network state
 * @param {number} learningRate - Learning rate for gradient descent
 */
function updateNetworkParameters(context, network, learningRate) {
    console.assert(context.gl !== null, 'WebGL context must exist');
    console.assert(network !== null, 'network must exist');
    console.assert(network.layers.length > 0, 'network must have at least one layer');
    console.assert(typeof learningRate === 'number' && learningRate > 0, 'learningRate must be positive number');

    for (let layerIndex = 0; layerIndex < network.layers.length; layerIndex++) {
        updateLayerWeights(context, layerIndex, network, learningRate);
        updateLayerBiases(context, layerIndex, network, learningRate);
    }
}

//////////////
// Training //
//////////////

/***************************
 * computeMeanSquaredError *
 ***************************
 *
 * Computes mean squared error between output and target vectors.
 * MSE = (1/n) * sum((output[i] - target[i])^2)
 * @param {Float32Array|Array<number>} output - Network output
 * @param {Float32Array|Array<number>} target - Target values
 * @returns {number} Mean squared error
 */
function computeMeanSquaredError(output, target) {
    console.assert(output !== null && output.length > 0, 'output must be non-empty');
    console.assert(target !== null && target.length > 0, 'target must be non-empty');
    console.assert(output.length === target.length, 'output and target must have same length');

    let sum = 0.0;
    for (let i = 0; i < output.length; i++) {
        const error = output[i] - target[i];
        sum += error * error;
    }
    return sum / output.length;
}

/**************************
 * trainnetworkSingleStep *
 **************************
 *
 * Performs a single training step: forward pass, backpropagation, and parameter update.
 * @param {Object} context - Graphics context
 * @param {Object} network - Network state
 * @param {Array<number>|Float32Array} inputValues - Input vector
 * @param {Array<number>|Float32Array} targetValues - Target output vector
 * @param {number} learningRate - Learning rate for gradient descent
 * @returns {number} Mean squared error for this training step
 */
function trainNetworkSingleStep(context, network, inputValues, targetValues, learningRate) {
    console.assert(context.gl !== null, 'WebGL context must exist');
    console.assert(network !== null, 'network must exist');
    console.assert(network.layers.length > 0, 'network must have at least one layer');
    console.assert(inputValues !== null && inputValues.length > 0, 'inputValues must be non-empty');
    console.assert(targetValues !== null && targetValues.length > 0, 'targetValues must be non-empty');
    console.assert(typeof learningRate === 'number' && learningRate > 0, 'learningRate must be positive number');

    const gl = context.gl;

    const output = computeNetworkForward(context, network, inputValues);

    if (!network.targetTexture || network.targetDimension !== targetValues.length) {
        network.targetDimension = targetValues.length;
        network.targetTexture = createTexture(context, targetValues.length, 1);
    }

    uploadTextureData(context, network.targetTexture, targetValues, targetValues.length, 1);

    computeNetworkBackpropagation(context, network);
    updateNetworkParameters(context, network, learningRate);

    return computeMeanSquaredError(output, targetValues);
}

/////////////
// Exports //
/////////////

export {
    initializeGraphicsContext,
    initializeNetwork,
    createLayerSpecification,
    createNetworkState,
    addLayerToNetwork,
    computeNetworkForward,
    trainNetworkSingleStep,
    applySigmoid,
    applyRelu,
    applyTanh,
    applyLinear
};