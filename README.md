# Voxy Engine

Voxy Engine is a raymarched mini voxel engine / game project. 

## Features
- A dynamic, falling sand simulated world
- Transparent and textured materials
- Smooth per voxel normals
- Raytraced ambient occlusion
- World editing at different block scales
- Terrain generation
- Player controller with world collisions
- Simple OpenGL engine to handle shaders, meshes, cameras, windowing, and math.

## Building

The project can be built by simply installing the .NET SDK: https://dotnet.microsoft.com/en-us/download/dotnet
The project is currently using .NET 8.0, however I will probably switch to .NET 9.0 soon.

Right now I am handling paths to the shader files by setting the working directory to the root of my project inside C#. This is obviously not a great way to do it, but for testing I haven't made a better solution.
So in order to run it, update this line in `Program.cs`
```csharp
Environment.CurrentDirectory = @"C:\Main\Projects\Coding\C#\VoxyEngine\src";
```

Then open the project and run `dotnet run` in the terminal, or use VSCode to debug the project.

## Screenshots

![image](https://github.com/user-attachments/assets/a3f1d0b1-31e4-4687-8482-e79f02436887)
![image](https://github.com/user-attachments/assets/a2f4f563-2c18-4c77-9832-03b069341d8a)
![image](https://github.com/user-attachments/assets/c4474e70-0fba-400f-b45b-b788143ee649)
![image](https://github.com/user-attachments/assets/78e2dd56-b4de-47ff-8a44-367a0da99f69)
![image](https://github.com/user-attachments/assets/4e66f017-f9ab-4e38-9749-d5230ff1b7f3)
