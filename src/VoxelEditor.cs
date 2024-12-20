using System.Numerics;
using System.Runtime.InteropServices;
using Silk.NET.OpenGL;
using Engine;

[StructLayout(LayoutKind.Sequential)]
public struct ClearResult
{
    public uint MaterialId;
    public uint Count;
}

public class VoxelEditorSystem : IDisposable
{
    private readonly GLContext ctx;
    private readonly ChunkManager chunkManager;
    private ComputeShader editShader;
    
    // Buffer for clear results
    private uint clearResultBuffer;
    private const int MAX_MATERIALS = 255;
    
    public VoxelEditorSystem(GLContext context, ChunkManager chunkManager)
    {
        this.ctx = context;
        this.chunkManager = chunkManager;
        
        // Create compute shader
        editShader = new ComputeShader(ctx, "shaders/voxel-edit.comp.glsl");
        
        // Initialize buffer
        CreateBuffers();
    }
    
    private unsafe void CreateBuffers()
    {
        ctx.ExecuteCmd((dt, gl) =>
        {
            // Create clear results buffer
            gl.GenBuffers(1, out clearResultBuffer);
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, clearResultBuffer);
            gl.BufferData(BufferTargetARB.ShaderStorageBuffer, 
                (nuint)(MAX_MATERIALS * sizeof(ClearResult)), 
                null, 
                BufferUsageARB.DynamicRead);
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, 0);
        });
    }
    
    /// <summary>
    /// Places voxels in a cubic region
    /// </summary>
    /// <param name="cornerPos">Voxel-space coordinate of the minimum corner of the cube</param>
    /// <param name="size">Size of the cube in voxels (max 64 in any dimension)</param>
    /// <param name="materialId">Material ID to place</param>
    public unsafe void PlaceVoxels(Vector3 cornerPos, Vector3 size, int materialId)
    {
        if (!ctx.IsCurrent)
            throw new InvalidOperationException("PlaceVoxels must be called from OpenGL thread");
            
        var gl = ctx.gl;
        
        // Ensure chunk manager buffers are bound
        chunkManager.BindBuffers(gl);
        
        // Set uniforms and dispatch shader
        editShader.Use();
        editShader.SetUniform("cornerPos", cornerPos);
        editShader.SetUniform("cubeSize", size);
        editShader.SetUniform("materialId", materialId);
        editShader.SetUniform("clear", false);
        
        uint groupsX = ((uint)size.X + 7) / 8;
        uint groupsY = ((uint)size.Y + 7) / 8;
        uint groupsZ = ((uint)size.Z + 7) / 8;
        editShader.Dispatch(groupsX, groupsY, groupsZ);
    }

    /// <summary>
    /// Clears voxels in a cubic region
    /// </summary>
    /// <param name="cornerPos">Voxel-space coordinate of the minimum corner of the cube</param>
    /// <param name="size">Size of the cube in voxels (max 64 in any dimension)</param>
    /// <returns>Array of clear results indicating what materials were cleared and their counts</returns>
    public unsafe ClearResult[] ClearVoxels(Vector3 cornerPos, Vector3 size)
    {
        if (!ctx.IsCurrent)
            throw new InvalidOperationException("ClearVoxels must be called from OpenGL thread");
            
        var gl = ctx.gl;
        
        // Clear the results buffer
        gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, clearResultBuffer);
        gl.ClearBufferData(GLEnum.ShaderStorageBuffer, 
            GLEnum.RG32ui,  // internal format
            GLEnum.RG,      // format
            GLEnum.UnsignedInt, 
            null);
        
        // Bind clear results buffer
        gl.BindBufferBase(BufferTargetARB.ShaderStorageBuffer, 0, clearResultBuffer);
        
        // Ensure chunk manager buffers are bound
        chunkManager.BindBuffers(gl);
        
        // Set uniforms and dispatch shader
        editShader.Use();
        editShader.SetUniform("cornerPos", cornerPos); 
        editShader.SetUniform("cubeSize", size);
        editShader.SetUniform("materialId", 0); // Ignored when clearing
        editShader.SetUniform("clear", true);
        
        uint groupsX = ((uint)size.X + 7) / 8;
        uint groupsY = ((uint)size.Y + 7) / 8;
        uint groupsZ = ((uint)size.Z + 7) / 8;
        editShader.Dispatch(groupsX, groupsY, groupsZ);
        
        // Read back the results
        Span<ClearResult> results = new ClearResult[MAX_MATERIALS];
        gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, clearResultBuffer);
        gl.GetBufferSubData(GLEnum.ShaderStorageBuffer, 0, 
            (nuint)(MAX_MATERIALS * sizeof(ClearResult)), results);
            
        // Trim the list to the number of materials actually cleared
        var resultsList = results.ToArray().ToList();
        return [.. resultsList.TakeWhile(r => r.Count > 0)];
    }
    
    public void Dispose()
    {
        ctx.ExecuteCmd((dt, gl) =>
        {
            editShader?.Dispose();
            gl.DeleteBuffer(clearResultBuffer);
        });
    }
}