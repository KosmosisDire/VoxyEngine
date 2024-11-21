
using System.Numerics;
using Engine;
using ImGuiNET;
using Silk.NET.Input;
using Silk.NET.OpenGL.Extensions.ImGui;
using VoxelEngine;


Environment.CurrentDirectory = @"C:\Main Documents\Projects\Coding\C#\VoxelRaymarcher\src";
var window = new Window();
await window.Load();


// lock cursor
var mouse = window.context.Input.Mice[0];
var keyboard = window.context.Input.Keyboards[0];
mouse.Click += (m, b, v) => mouse.Cursor.CursorMode = CursorMode.Disabled;
var camera = new FreeLookCamera(keyboard, mouse, new Vector3(1, 0.2f, 1) * (ChunkManager2.chunkGridWidth + 1), new Vector3(0, 0, 0));
camera.speed = 10;

ImGuiController imgui = null;
window.context.ExecuteCmd((dt, gl) =>
{
    imgui = new ImGuiController(gl, window.SilkWindow, window.context.Input);
});


var voxelRenderer = new VoxelRenderer2(window, camera);

window.OnRender((dt) =>
{
    if (imgui == null) return;

    voxelRenderer.Draw(dt);
    
    // show imgui fps
    ImGui.Text($"Frame Time: {((1/window.FPS) * 1000)}ms");
    // show camera position
    ImGui.Text($"Camera Position: {camera.Position}");
    imgui.Render();
});

window.OnUpdate((dt) =>
{
    if (imgui == null) return;

    imgui.Update((float)dt);
    camera.AspectRatio = (float)window.Size.X / window.Size.Y;
    camera.Update(dt);
});