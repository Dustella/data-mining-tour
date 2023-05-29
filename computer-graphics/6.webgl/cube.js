var canvas;
var gl;

var NumVertices = 36;

var points = [];
var colors = [];

var xAxis = 0;
var yAxis = 1;
var zAxis = 2;

var axis = 0;
var theta = [0, 0, 0];

var thetaLoc;

var flag = true;

window.onload = function init() {
  canvas = document.getElementById("gl-canvas");

  gl = WebGLUtils.setupWebGL(canvas);
  if (!gl) {
    alert("WebGL isn't available");
  }

  colorCube();

  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(1.0, 1.0, 1.0, 1.0);

  gl.enable(gl.DEPTH_TEST);

  //
  //  Load shaders and initialize attribute buffers
  //
  var program = initShaders(gl, "vertex-shader", "fragment-shader");
  gl.useProgram(program);

  var cBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, cBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, flatten(colors), gl.STATIC_DRAW);

  var vColor = gl.getAttribLocation(program, "vColor");
  gl.vertexAttribPointer(vColor, 4, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(vColor);

  var vBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, flatten(points), gl.STATIC_DRAW);

  var vPosition = gl.getAttribLocation(program, "vPosition");
  gl.vertexAttribPointer(vPosition, 4, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(vPosition);

  thetaLoc = gl.getUniformLocation(program, "theta");

  //event listeners for buttons

  document.getElementById("xButton").onclick = function () {
    axis = xAxis;
  };
  document.getElementById("yButton").onclick = function () {
    axis = yAxis;
  };
  document.getElementById("zButton").onclick = function () {
    axis = zAxis;
  };
  document.getElementById("ButtonT").onclick = function () {
    flag = !flag;
  };

  render();
};

function colorCube() {
  const faceColors = [
    [0.0, 1.0, 1.0, 1.0], // Front face: orange
    [1.0, 0.0, 0.0, 1.0], // Back face: red
    [0.0, 1.0, 0.0, 1.0], // Top face: green
    [0.0, 0.0, 1.0, 1.0], // Bottom face: blue
    [1.0, 1.0, 0.0, 1.0], // Right face: yellow
    [1.0, 0.0, 1.0, 1.0], // Left face: purple
  ];

  //   these points should be three dim , triangles are 3 points

  //   points = indices;
  points = generateCubeCoordinates();
  console.log(points);

  // colors = faceColors;
  for (var j = 0; j < faceColors.length; ++j) {
    const c = faceColors[j];
    // Repeat each color four times for the four vertices of the face
    colors = colors.concat(c, c, c);
    colors = colors.concat(c, c, c);
  }
}

function render() {
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  if (flag) theta[axis] += 2.0;
  gl.uniform3fv(thetaLoc, theta);

  gl.drawArrays(gl.TRIANGLES, 0, NumVertices);

  requestAnimFrame(render);
}

function generateCubeCoordinates() {
  const size = 0.5;
  const vertices = [
    [-size, -size, -size, 1],
    [size, -size, -size, 1],
    [size, size, -size, 1],
    [-size, size, -size, 1],
    [-size, -size, size, 1],
    [size, -size, size, 1],
    [size, size, size, 1],
    [-size, size, size, 1],
  ];

  const faces = [
    [0, 1, 2, 3],
    [1, 5, 6, 2],
    [5, 4, 7, 6],
    [4, 0, 3, 7],
    [3, 2, 6, 7],
    [4, 5, 1, 0],
  ];

  const coordinates = [];

  for (let i = 0; i < faces.length; i++) {
    const face = faces[i];
    const triangle1 = [vertices[face[0]], vertices[face[1]], vertices[face[2]]];
    const triangle2 = [vertices[face[0]], vertices[face[2]], vertices[face[3]]];
    coordinates.push([triangle1, triangle2]);
  }

  return coordinates.flat(2);
}
