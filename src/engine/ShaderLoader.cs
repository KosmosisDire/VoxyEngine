using System.Text.RegularExpressions;
using Silk.NET.OpenGL;
namespace Engine;

public enum ShaderType
{
    Vertex = GLEnum.VertexShader,
    Fragment = GLEnum.FragmentShader,
    Compute = GLEnum.ComputeShader
}

public class ShaderLoader : IDisposable
{
    private string source;
    public uint handle { get; private set; }
    private GL gl;
    private HashSet<string> processedIncludes = new();
    private HashSet<string> processedExtensions = new();
    private string versionDirective = null;

    public ShaderLoader(GL gl, string shaderPath, ShaderType type)
    {
        this.gl = gl;
        if (!File.Exists(shaderPath))
        {
            throw new FileNotFoundException($"Shader file not found: {shaderPath}");
        }

        source = File.ReadAllText(shaderPath);
        source = StripComments(source);  // Strip all comments first
        source = LoadIncludes(source, Path.GetFullPath(shaderPath));
        source = ProcessDirectives(source);
        source = source.Trim();
        
        handle = gl.CreateShader((GLEnum)type);
        gl.ShaderSource(handle, source);
        gl.CompileShader(handle);

        // Check for compilation errors
        gl.GetShader(handle, GLEnum.CompileStatus, out int status);
        if (status == 0)
        {
            string infoLog = gl.GetShaderInfoLog(handle);
            var lines = source.Split('\n');
            var errorMatch = Regex.Match(infoLog.Split('\n')[0], @"\((\d+)\)");
            
            if (errorMatch.Success)
            {
                var line = int.Parse(errorMatch.Groups[1].Value);
                Console.WriteLine($"Error compiling shader {shaderPath}: {infoLog}");
                if (line > 0 && line <= lines.Length)
                {
                    Console.WriteLine($"Line {line}: {lines[line - 1]}");
                }
            }
            else
            {
                Console.WriteLine($"Error compiling shader {shaderPath}: {infoLog}");
            }
        }
    }

    private string StripComments(string source)
    {
        // Remove multi-line comments first
        source = Regex.Replace(source, @"/\*[\s\S]*?\*/", "", RegexOptions.Multiline);
        
        // Remove single-line comments
        source = Regex.Replace(source, @"//.*$", "", RegexOptions.Multiline);
        
        // Remove empty lines and trim
        return string.Join("\n", 
            source.Split('\n')
                 .Where(line => !string.IsNullOrWhiteSpace(line))
                 .Select(line => line.Trim()));
    }

    private string LoadIncludes(string sourceCode, string currentFilePath)
    {
        var currentDir = Path.GetDirectoryName(currentFilePath);
        var includeRegex = new Regex(@"#include\s+""(.+)""");
        var matches = includeRegex.Matches(sourceCode);
        
        // Process includes in reverse order to maintain correct positions
        for (int i = matches.Count - 1; i >= 0; i--)
        {
            var include = matches[i];
            string includePath = include.Groups[1].Value;
            string fullIncludePath = ResolvePath(includePath, currentDir);

            // Always remove the include directive
            sourceCode = sourceCode.Remove(include.Index, include.Length);

            // Check for circular includes
            if (processedIncludes.Contains(fullIncludePath))
            {
                Console.WriteLine($"Warning: Circular include detected for {includePath}, skipping...");
                continue;
            }

            if (!File.Exists(fullIncludePath))
            {
                throw new FileNotFoundException($"Include file not found: {fullIncludePath} (included from {currentFilePath})");
            }

            processedIncludes.Add(fullIncludePath);
            string includeSource = File.ReadAllText(fullIncludePath);

            // Strip comments and process includes in the included file
            includeSource = StripComments(includeSource);
            includeSource = LoadIncludes(includeSource, fullIncludePath);
            includeSource = includeSource.Trim();

            // Insert the processed content where the include directive was
            sourceCode = sourceCode.Insert(include.Index, includeSource);
        }

        return sourceCode;
    }

    private string ProcessDirectives(string sourceCode)
    {
        var lines = sourceCode.Split('\n');
        var processedLines = new List<string>();
        
        foreach (var line in lines)
        {
            string trimmedLine = line.Trim();
            
            // Handle version directive
            if (trimmedLine.StartsWith("#version"))
            {
                if (versionDirective == null)
                {
                    versionDirective = trimmedLine;
                    processedLines.Insert(0, versionDirective); // Always put version at the top
                }
                continue; // Skip this line as we've either used it or it's a duplicate
            }
            
            // Handle extension directive
            if (trimmedLine.StartsWith("#extension"))
            {
                if (!processedExtensions.Contains(trimmedLine))
                {
                    processedExtensions.Add(trimmedLine);
                    // Add extensions right after the version directive
                    int insertIndex = versionDirective != null ? 1 : 0;
                    processedLines.Insert(insertIndex, trimmedLine);
                }
                continue;
            }
            
            processedLines.Add(line); // Keep all other lines as is
        }
        
        return string.Join("\n", processedLines);
    }

    private string ResolvePath(string includePath, string currentDir)
    {
        // Handle absolute paths
        if (Path.IsPathRooted(includePath))
        {
            return includePath;
        }

        // Handle relative paths
        string resolvedPath = Path.GetFullPath(Path.Combine(currentDir, includePath));
        
        // Verify the resolved path is valid and normalized
        try
        {
            resolvedPath = Path.GetFullPath(resolvedPath);
            return resolvedPath;
        }
        catch (Exception ex)
        {
            throw new ArgumentException($"Invalid include path: {includePath}", ex);
        }
    }

    public void Dispose()
    {
        gl.DeleteShader(handle);
        GC.SuppressFinalize(this);
    }
}