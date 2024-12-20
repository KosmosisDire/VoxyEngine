using System.Numerics;
using Engine;
using ImGuiNET;
using Silk.NET.Input;
using Silk.NET.OpenGL.Extensions.ImGui;

// Set working directory to find shaders
Environment.CurrentDirectory = @"C:\Main\Projects\Coding\C#\VoxelRaymarcher\src";

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
var player = new PlayerController(
    keyboard, 
    mouse,
    new Vector3(1, 1, 1) * (ChunkManager.GridSize / 2),
    new Vector3(0, 0, 0)
);

// Initialize ImGui
ImGuiController imgui = null;
window.context.ExecuteCmd((dt, gl) =>
{
    imgui = new ImGuiController(gl, window.SilkWindow, window.context.Input);
});

// Create voxel renderer
var voxelRenderer = new VoxelRenderer(window, player.Camera);
player.SetVoxelWorld(voxelRenderer);

List<float> frameTimes = new List<float>();
float averageDt = 16;

// Render loop
window.OnRender((dt) =>
{
    if (imgui == null) return;

    player.Update(dt);
    voxelRenderer.Draw(dt);

    averageDt = averageDt * 0.95f + (float)dt * 0.05f;
    
    // ImGui overlay 
    ImGui.Begin("Debug Info", ImGuiWindowFlags.AlwaysAutoResize);
    ImGui.Text($"Player Position: {player.Position}");
    ImGui.Text($"Grounded: {player.IsGrounded}");
    ImGui.Text($"Step Height: {player.currentStepHeight}");
    ImGui.Text($"Hovered Material: {Materials.GetMaterialName(voxelRenderer.hoveredCastResult.materialId)}");
    ImGui.Text($"Selected Material: {Materials.GetMaterialName(voxelRenderer.selectedMaterial)}");
    ImGui.Text($"Place Size: {voxelRenderer.placeSize}");
    ImGui.Separator();
    ImGui.Text($"FPS: {1.0/averageDt:F1}");
    ImGui.Text($"Frame Time: {averageDt * 1000:F1}ms");
    frameTimes.Add((float)averageDt * 1000);
    if (frameTimes.Count > 1000)
    {
        frameTimes.RemoveAt(0);
    }
    
    var values = frameTimes.ToArray();
    ImGui.PlotLines("Frame Times", ref values[0], frameTimes.Count, 0, "Frame Time (ms)");
    ImGui.End();
    
    imgui.Render();
});

// Update loop
window.OnUpdate((dt) =>
{
    if (imgui == null) return;

    imgui.Update((float)dt);
    player.Camera.AspectRatio = (float)window.Size.X / window.Size.Y;
});