using System.Numerics;
using System.Runtime.InteropServices;
using Silk.NET.OpenGL;
using Engine;

[StructLayout(LayoutKind.Explicit)]
public struct RaycastHit
{
    [FieldOffset(0)] public Vector3 position;
    [FieldOffset(16)] public Vector3 cellPosition;
    [FieldOffset(32)] public Vector3 normal;
    [FieldOffset(48)] public Vector3 perVoxNormal;
    [FieldOffset(64)] public Vector3 color;
    [FieldOffset(76)] public bool valid;
    [FieldOffset(80)] public uint materialId;
    [FieldOffset(84)] public float distance;
    [FieldOffset(88)] public int blockIndex;
    [FieldOffset(92)] public int voxelIndex;

    public RaycastHit()
    {
        position = Vector3.Zero;
        cellPosition = Vector3.Zero;
        normal = Vector3.Zero;
        perVoxNormal = Vector3.Zero;
        color = Vector3.Zero;
        valid = false;
        materialId = 0;
        distance = 0;
        blockIndex = 0;
        voxelIndex = 0;
    }
}

// Original managed version that we'll use in the rest of the codebase
public struct RaycastResult
{
    public Vector3 accumulatedColor;
    public float opacity;
    public int numHits;
    public int totalSteps;
    public bool hasHit;
    public int hitCount;
    public RaycastHit[] hits;  // Managed array

    public RaycastResult()
    {
        accumulatedColor = Vector3.Zero;
        opacity = 0;
        numHits = 0;
        totalSteps = 0;
        hasHit = false;
        hitCount = 0;
        hits = new RaycastHit[3];  // MaxHits = 3
    }
}

// Unmanaged version for GPU buffer transfers
[StructLayout(LayoutKind.Explicit, Size = 320)]  // 32 + (96 * 3) = 320 bytes
public unsafe struct UnmanagedRaycastResult
{
    [FieldOffset(0)] public Vector3 accumulatedColor;
    [FieldOffset(12)] public float opacity;
    [FieldOffset(16)] public int numHits;
    [FieldOffset(20)] public int totalSteps;
    [FieldOffset(24)] public bool hasHit;
    [FieldOffset(28)] public int hitCount;

    // Fixed size array of hits embedded directly in the struct
    [FieldOffset(32)] public fixed byte hits[288];  // 96 bytes * 3 hits = 288 bytes

    // Convert to managed version
    public RaycastResult ToManaged()
    {
        var result = new RaycastResult
        {
            accumulatedColor = accumulatedColor,
            opacity = opacity,
            numHits = numHits,
            totalSteps = totalSteps,
            hasHit = hasHit,
            hitCount = hitCount,
            hits = new RaycastHit[3]
        };

        // Copy each hit
        fixed (byte* srcHits = hits)
        {
            for (int i = 0; i < 3; i++)
            {
                RaycastHit* srcHit = (RaycastHit*)(srcHits + (i * 96));  // 96 is size of RaycastHit
                result.hits[i] = *srcHit;
            }
        }

        return result;
    }
}

[StructLayout(LayoutKind.Explicit)]
public struct RaycastInput
{
    [FieldOffset(0)] public Vector3 origin;       // 12 bytes
    [FieldOffset(12)] public float padding1;       // 4 bytes for alignment
    [FieldOffset(16)] public Vector3 direction;    // 12 bytes
    [FieldOffset(28)] public float maxDistance;    // 4 bytes

    public RaycastInput(Vector3 origin, Vector3 direction, float maxDistance)
    {
        this.origin = origin;
        this.direction = direction;
        this.maxDistance = maxDistance;
        this.padding1 = 0;
    }
}

public class RaycastSystem : IDisposable
{
    private readonly GLContext ctx;
    private readonly ChunkManager chunkManager;
    private ComputeShader raycastShader;

    private uint rayBuffer;
    private uint resultBuffer;
    private const int MAX_BATCH_SIZE = 1024;

    public RaycastSystem(GLContext context, ChunkManager chunkManager)
    {
        this.ctx = context;
        this.chunkManager = chunkManager;

        raycastShader = new ComputeShader(ctx, "shaders/raycast.comp.glsl");
        CreateBuffers();
    }

    private unsafe void CreateBuffers()
    {
        ctx.ExecuteCmd((dt, gl) =>
        {
            gl.GenBuffers(1, out rayBuffer);
            gl.GenBuffers(1, out resultBuffer);

            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, rayBuffer);
            gl.BufferData(BufferTargetARB.ShaderStorageBuffer,
                (nuint)(MAX_BATCH_SIZE * sizeof(RaycastInput)),
                null,
                BufferUsageARB.DynamicDraw);

            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, resultBuffer);
            gl.BufferData(BufferTargetARB.ShaderStorageBuffer,
                (nuint)(MAX_BATCH_SIZE * Marshal.SizeOf<RaycastResult>()),
                null,
                BufferUsageARB.DynamicDraw);

            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, 0);
        });
    }

    /// <summary>
    /// Single raycast - only valid when called from OpenGL thread.
    /// Returns full raycast result including all hits and accumulated data.
    /// </summary>
    public unsafe RaycastResult Raycast(Vector3 origin, Vector3 direction, float maxDistance = 64)
    {
        var input = new RaycastInput(origin, direction, maxDistance);
        return RaycastBatch(new[] { input })[0];
    }

    public unsafe Span<RaycastResult> RaycastBatch(RaycastInput[] inputs)
    {
        if (!ctx.IsCurrent)
            throw new InvalidOperationException("Raycast must be called from OpenGL thread");

        if (inputs.Length > MAX_BATCH_SIZE)
            throw new ArgumentException($"Batch size exceeds maximum of {MAX_BATCH_SIZE}");

        if (inputs.Length == 0)
            return [];

        // Allocate unmanaged results array
        Span<UnmanagedRaycastResult> unmanagedResults = new UnmanagedRaycastResult[inputs.Length];
        RaycastResult[] results = new RaycastResult[inputs.Length];

        var gl = ctx.gl;

        // Upload ray data
        gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, rayBuffer);
        gl.BufferSubData(BufferTargetARB.ShaderStorageBuffer, 0,
            (nuint)(inputs.Length * sizeof(RaycastInput)),
            [.. inputs]);

        // Bind buffers for compute shader
        gl.BindBufferBase(BufferTargetARB.ShaderStorageBuffer, 9, rayBuffer);
        gl.BindBufferBase(BufferTargetARB.ShaderStorageBuffer, 10, resultBuffer);

        // Ensure chunk manager buffers are bound
        chunkManager.BindBuffers(gl);

        // Dispatch compute shader
        raycastShader.Use();
        raycastShader.SetUniform("rayCount", inputs.Length);
        uint groupCount = ((uint)inputs.Length + 63) / 64;
        raycastShader.Dispatch(groupCount, 1, 1);

        // Read back results into unmanaged buffer
        gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, resultBuffer);
        gl.GetBufferSubData(GLEnum.ShaderStorageBuffer, 0, (nuint)(inputs.Length * sizeof(UnmanagedRaycastResult)), unmanagedResults);
        
        // Convert to managed results
        for (int i = 0; i < inputs.Length; i++)
        {
            results[i] = unmanagedResults[i].ToManaged();
        }

        return results;
    }

    public void Dispose()
    {
        ctx.ExecuteCmd((dt, gl) =>
        {
            raycastShader?.Dispose();
            gl.DeleteBuffer(rayBuffer);
            gl.DeleteBuffer(resultBuffer);
        });
    }
}