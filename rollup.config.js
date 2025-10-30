export default [
    // Browser IIFE build
    {
        input: 'src/neural-network.js',
        output: {
            file: 'dist/neural-network.js',
            format: 'iife',
            name: 'NeuralNetwork',
            exports: 'named'
        }
    }
];
