using System.Numerics;
using Engine;
using ImGuiNET;
using Silk.NET.Input;
using Silk.NET.OpenGL.Extensions.ImGui;

// Set working directory to find shaders
Environment.CurrentDirectory = @"C:\Main Documents\Projects\Coding\C#\VoxelRaymarcher\src";

// Initialize window and input
var window = new Window();
await window.Load();

// Setup input handlers
var mouse = window.context.Input.Mice[0];
var keyboard = window.context.Input.Keyboards[0];

// Lock cursor on click
mouse.Click += (m, b, v) => mouse.Cursor.CursorMode = CursorMode.Disabled;

// Initialize camera
// Position camera based on world size for better initial view
float worldSize = ChunkManager2.GridSize * ChunkManager2.ChunkSize;
var camera = new FreeLookCamera(
    keyboard, 
    mouse,
    new Vector3(1, 0.2f, 1) * (ChunkManager2.GridSize + 1),
    new Vector3(0, 0, 0)
);
camera.speed = worldSize * 0.03f; // Scale camera speed with world size

// Initialize ImGui
ImGuiController imgui = null;
window.context.ExecuteCmd((dt, gl) =>
{
    imgui = new ImGuiController(gl, window.SilkWindow, window.context.Input);
});

// Create voxel renderer
var voxelRenderer = new VoxelRenderer2(window, camera);

// Render loop
window.OnRender((dt) =>
{
    if (imgui == null) return;

    voxelRenderer.Draw(dt);
    
    // ImGui overlay
    ImGui.Begin("Debug Info", ImGuiWindowFlags.AlwaysAutoResize);
    ImGui.Text($"FPS: {1.0/dt:F1}");
    ImGui.Text($"Frame Time: {dt * 1000:F1}ms");
    ImGui.Text($"Camera Position: {camera.Position}");
    ImGui.End();
    
    imgui.Render();
});

// Update loop
window.OnUpdate((dt) =>
{
    if (imgui == null) return;

    imgui.Update((float)dt);
    camera.AspectRatio = (float)window.Size.X / window.Size.Y;
    camera.Update(dt);

    // ESC to exit
    if (keyboard.IsKeyPressed(Key.Escape))
    {
        window.SilkWindow.Close();
    }
});