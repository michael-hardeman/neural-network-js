# Neural Network Library - Build Instructions

## Quick Start

1. Install dependencies:
```bash
npm install
```

2. Build the library:
```bash
npm run build
```

This creates two files in the `dist/` directory:
- `src/neural-network.js` - ES module for use with bundlers or modern browsers
- `dist/neural-network.js` - Browser-ready IIFE (Immediately Invoked Function Expression)

## Using the Build

### Browser (IIFE)
```html
<script src="dist/neural-network.js"></script>
<script>
    const { initializeGraphicsContext, initializeNetwork } = NeuralNetwork;

    const context = initializeGraphicsContext();
    const network = initializeNetwork(context, [...]);
</script>
```

### ES Module
Technically this is a module and you could attempt to use it in nodejs
but currently it does not function as there are no GLES (WebGL 2.0)
compatible npm packages. I have no plans currently to make this work.

```javascript
import {
    initializeGraphicsContext,
    initializeNetwork,
    createLayerSpecification
} from './src/neural-network.js';

const context = initializeGraphicsContext();
```

## Development

Watch mode (rebuilds on file changes):
```bash
npm run watch
```

## Testing

Open `examples/browser.html` in a browser after building. The test page uses the browser IIFE build.
