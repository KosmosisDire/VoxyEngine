using Silk.NET.OpenGL;

namespace Engine;

public class ComputeShader : BaseShader
{
    private bool isValid = false;

    public ComputeShader(GLContext ctx, string computeShaderPath) : base(ctx)
    {
        ctx.ExecuteCmd((dt, gl) =>
        {
            InitializeShader(new ShaderLoader(ctx, computeShaderPath, ShaderType.ComputeShader));
            LinkProgram();

            // Enable debug output for compute shaders
            gl.Enable(EnableCap.DebugOutput);
            gl.Enable(EnableCap.DebugOutputSynchronous);
            
            // Validate the program specifically for compute
            gl.ValidateProgram(handle);
            gl.GetProgram(handle, GLEnum.ValidateStatus, out int validateStatus);
            if (validateStatus == 0)
            {
                string validateLog = gl.GetProgramInfoLog(handle);
                Console.WriteLine($"Compute shader validation failed: {validateLog}");
                isValid = false;
            }
            else
            {
                isValid = true;
                Console.WriteLine("Compute shader validated successfully");
            }
        });
    }
 
    public void Dispatch(uint groupsX, uint groupsY = 1, uint groupsZ = 1)
    {
        if (!isValid)
        {
            Console.WriteLine("Cannot dispatch - compute shader is not valid");
            return;
        }

        if (groupsX == 0 || groupsY == 0 || groupsZ == 0)
        {
            throw new ArgumentException("Dispatch group size cannot be zero");
        }
        
        ctx.ExecuteCmd((dt, gl) =>
        {
            // Clear any previous errors
            while (gl.GetError() != GLEnum.NoError) 
            {
                if (gl.GetError() == GLEnum.NoError) break;
                Console.WriteLine($"Clearing previous error: {gl.GetError()}");
            }
            
            Use();
            
            // Check if shader program is current
            gl.GetInteger(GLEnum.CurrentProgram, out int currentProgram);
            if ((uint)currentProgram != handle)
            {
                Console.WriteLine($"Warning: Wrong program bound. Expected {handle}, got {currentProgram}");
                return;
            }

            // Check program link status
            gl.GetProgram(handle, GLEnum.LinkStatus, out int linkStatus);
            if (linkStatus == 0)
            {
                Console.WriteLine("Program is not linked!");
                string infoLog = gl.GetProgramInfoLog(handle);
                Console.WriteLine($"Link info: {infoLog}");
                return;
            }
            
            // Full barrier to ensure previous operations are complete
            gl.MemoryBarrier(MemoryBarrierMask.AllBarrierBits);
            
            gl.DispatchCompute(groupsX, groupsY, groupsZ);
            
            // Wait for compute shader to finish
            gl.MemoryBarrier(MemoryBarrierMask.AllBarrierBits);
            
            // Check for errors
            var error = gl.GetError();
            if (error != GLEnum.NoError)
            {
                Console.WriteLine($"OpenGL error after compute dispatch: {error}");
                // Dump some debug info
                gl.GetInteger(GLEnum.CurrentProgram, out int program);
                gl.GetProgram(handle, GLEnum.ActiveUniforms, out int activeUniforms);
                gl.GetProgram(handle, GLEnum.ActiveUniformBlocks, out int activeUniformBlocks);
                Console.WriteLine($"Current program: {program}");
                Console.WriteLine($"Active uniforms: {activeUniforms}");
                Console.WriteLine($"Active uniform blocks: {activeUniformBlocks}");
                Console.WriteLine($"Dispatch size: {groupsX}, {groupsY}, {groupsZ}");
            }
        });
    }
}