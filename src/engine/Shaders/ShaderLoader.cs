using System.Text;
using System.Text.RegularExpressions;
using Silk.NET.OpenGL;

namespace Engine;

public record struct LineInfo
{
    public string FilePath { get; init; }
    public int OriginalLineNumber { get; init; }
    public string OriginalContent { get; init; }

    public LineInfo(string filePath, int lineNumber, string content)
    {
        FilePath = filePath;
        OriginalLineNumber = lineNumber;
        OriginalContent = content;
    }
}

public class ProcessedSource
{
    public string ProcessedContent { get; set; }
    public List<LineInfo> LineMapping { get; set; }

    public ProcessedSource(string content)
    {
        ProcessedContent = content;
        LineMapping = new List<LineInfo>();
    }

    public ProcessedSource(string content, List<LineInfo> mapping)
    {
        ProcessedContent = content;
        LineMapping = mapping;
    }

    public void InsertContent(int position, ProcessedSource source)
    {
        // Count the number of newlines before the insertion point
        int insertLine = ProcessedContent.Take(position).Count(c => c == '\n');

        // Insert the content
        ProcessedContent = ProcessedContent.Insert(position, source.ProcessedContent);

        // Make sure we don't try to insert beyond the end of our line mapping
        insertLine = Math.Min(insertLine, LineMapping.Count);

        // Insert the line mappings at the correct position
        if (source.LineMapping.Count > 0)
        {
            LineMapping.InsertRange(insertLine, source.LineMapping);
        }
    }

    public void RemoveLine(int lineNumber)
    {
        var lines = ProcessedContent.Split('\n');
        if (lineNumber < lines.Length)
        {
            var newLines = lines.Take(lineNumber).Concat(lines.Skip(lineNumber + 1));
            ProcessedContent = string.Join('\n', newLines);
            if (lineNumber < LineMapping.Count)
            {
                LineMapping.RemoveAt(lineNumber);
            }
        }
    }
}

public class ShaderError
{
    public string FilePath { get; set; }
    public int LineNumber { get; set; }
    public string LineContent { get; set; }
    public string ErrorMessage { get; set; }

    public override string ToString()
    {
        return $"Error in {FilePath} at line {LineNumber}:\n{LineContent}\n{ErrorMessage}";
    }
}

public class ShaderCompilationException : Exception
{
    public List<ShaderError> Errors { get; }

    public ShaderCompilationException(List<ShaderError> errors)
        : base(BuildErrorMessage(errors))
    {
        Errors = errors;
    }

    private static string BuildErrorMessage(List<ShaderError> errors)
    {
        return "Shader compilation failed with the following errors:\n" +
               string.Join("\n\n", errors.Select(e => e.ToString()));
    }
}

public class ShaderLoader : IDisposable
{
    private ProcessedSource processedSource;
    public uint handle { get; private set; }
    public ShaderType shaderType { get; private set; }
    private GL gl;
    private HashSet<string> processedIncludes = new();
    private HashSet<string> processedExtensions = new();
    private string versionDirective = null;
    private string rootDirectory;
    private string mainShaderPath;
    private Dictionary<string, FileSystemWatcher> fileWatchers = new();
    private readonly object recompileLock = new object();
    private bool needsRecompile = false;

    public event Action<List<ShaderError>> OnCompilationError;
    public event Action OnRecompiled;

    public ShaderLoader(GL gl, string shaderPath, ShaderType type)
    {
        this.gl = gl;
        this.mainShaderPath = Path.GetFullPath(shaderPath);
        this.shaderType = type;

        CompileInitial();
        SetupFileWatchers();
    }

    private void CompileInitial()
    {
        Console.WriteLine($"-----------------------------\nLoading shader: {mainShaderPath}");

        if (!File.Exists(mainShaderPath))
        {
            throw new FileNotFoundException($"Shader file not found: {mainShaderPath}");
        }

        rootDirectory = Path.GetDirectoryName(Path.GetFullPath(mainShaderPath));
        var source = File.ReadAllText(mainShaderPath);
        processedSource = CreateProcessedSource(mainShaderPath, source);

        ProcessShaderSource();
        CompileShader();
    }

    private ProcessedSource CreateProcessedSource(string filePath, string content)
    {
        var source = new ProcessedSource(content);
        var lines = content.Split('\n');
        for (int i = 0; i < lines.Length; i++)
        {
            source.LineMapping.Add(new LineInfo(filePath, i + 1, lines[i]));
        }
        return source;
    }

    private void ProcessShaderSource()
    {
        processedIncludes.Clear();
        processedExtensions.Clear();
        versionDirective = null;

        // Process in correct order: version, extensions, includes, other directives
        processedSource = ExtractVersionAndExtensions(processedSource);
        processedSource = StripComments(processedSource);
        processedSource = LoadIncludes(processedSource, mainShaderPath);
        processedSource = ProcessDirectives(processedSource);

        // Add version and extensions at the start
        var headerSource = new ProcessedSource("");
        if (versionDirective != null)
        {
            headerSource.ProcessedContent = versionDirective + "\n";
            headerSource.LineMapping.Add(new LineInfo(mainShaderPath, 0, versionDirective));
        }
        else
        {
            string defaultVersion = "#version 460";
            headerSource.ProcessedContent = defaultVersion + "\n";
            headerSource.LineMapping.Add(new LineInfo(mainShaderPath, 0, defaultVersion));
            Console.WriteLine("Warning: No version directive found, adding default #version 460");
        }

        foreach (var extension in processedExtensions)
        {
            headerSource.ProcessedContent += extension + "\n";
            headerSource.LineMapping.Add(new LineInfo(mainShaderPath, 0, extension));
        }

        // Combine header and processed content
        var finalSource = new ProcessedSource("");
        finalSource.ProcessedContent = headerSource.ProcessedContent + processedSource.ProcessedContent;
        finalSource.LineMapping.AddRange(headerSource.LineMapping);
        finalSource.LineMapping.AddRange(processedSource.LineMapping);
        processedSource = finalSource;

        // Write processed content to debug files
        var baseDir = Path.GetDirectoryName(mainShaderPath);
        var processedDir = Path.Combine(baseDir ?? "", "processed");
        Directory.CreateDirectory(processedDir);

        // Write the raw processed content
        var rawPath = Path.Combine(processedDir,
            Path.GetFileNameWithoutExtension(mainShaderPath) + ".raw.glsl");
        File.WriteAllText(rawPath, processedSource.ProcessedContent);

        // Write the annotated version with line mappings
        var annotatedPath = Path.Combine(processedDir,
            Path.GetFileNameWithoutExtension(mainShaderPath) + ".annotated.glsl");
        using (var writer = new StreamWriter(annotatedPath))
        {
            var processedLines = processedSource.ProcessedContent.Split('\n');
            for (int i = 0; i < Math.Max(processedLines.Length, processedSource.LineMapping.Count); i++)
            {
                if (i < processedSource.LineMapping.Count)
                {
                    var lineInfo = processedSource.LineMapping[i];
                    writer.WriteLine($"// [{Path.GetFileNameWithoutExtension(lineInfo.FilePath)}:{lineInfo.OriginalLineNumber}]");
                }

                if (i < processedLines.Length)
                {
                    writer.WriteLine(processedLines[i]);
                }
                else
                {
                    writer.WriteLine("// WARNING: Missing processed line");
                }
            }
        }
    }

    private ProcessedSource ExtractVersionAndExtensions(ProcessedSource source)
    {
        var lines = source.ProcessedContent.Split('\n');
        var processedLines = new List<string>();
        var newMapping = new List<LineInfo>();
        bool keepRemainingLines = false;

        for (int i = 0; i < lines.Length; i++)
        {
            string line = lines[i];
            string trimmedLine = line.Trim();

            if (trimmedLine.StartsWith("#version"))
            {
                if (versionDirective == null)
                {
                    versionDirective = trimmedLine;
                }
                keepRemainingLines = true;
                continue;
            }
            else if (trimmedLine.StartsWith("#extension"))
            {
                processedExtensions.Add(trimmedLine);
                continue;
            }

            if (keepRemainingLines)
            {
                processedLines.Add(line);
                if (i < source.LineMapping.Count)
                {
                    newMapping.Add(source.LineMapping[i]);
                }
            }
        }

        return new ProcessedSource(string.Join("\n", processedLines), newMapping);
    }

    private ProcessedSource StripComments(ProcessedSource source)
    {
        var processed = source.ProcessedContent;
        var lineMapping = new List<LineInfo>(source.LineMapping);

        // Remove multi-line comments
        var multiLineComments = Regex.Matches(processed, @"/\*[\s\S]*?\*/");
        foreach (Match match in multiLineComments.Reverse())
        {
            var commentedLines = processed.Substring(0, match.Index).Count(c => c == '\n');
            var commentLength = match.Value.Count(c => c == '\n');

            // Remove the corresponding lines from mapping
            if (commentedLines + commentLength < lineMapping.Count)
            {
                lineMapping.RemoveRange(commentedLines, commentLength);
            }

            processed = processed.Remove(match.Index, match.Length);
        }

        // Remove single-line comments but preserve empty lines for mapping
        var lines = processed.Split('\n');
        var newLines = new List<string>();
        var newMapping = new List<LineInfo>();

        for (int i = 0; i < lines.Length; i++)
        {
            var line = lines[i];
            var commentIndex = line.IndexOf("//");
            if (commentIndex >= 0)
            {
                line = line.Substring(0, commentIndex).TrimEnd();
            }

            newLines.Add(line);
            if (i < lineMapping.Count)
            {
                newMapping.Add(lineMapping[i]);
            }
        }

        return new ProcessedSource(string.Join("\n", newLines), newMapping);
    }

    private ProcessedSource LoadIncludes(ProcessedSource source, string currentFilePath)
    {
        var currentDir = Path.GetDirectoryName(currentFilePath);
        var includeRegex = new Regex(@"#include\s+""([^""]+)""");
        bool foundIncludes;

        do
        {
            foundIncludes = false;
            var content = source.ProcessedContent;
            var match = includeRegex.Match(content);

            if (match.Success)
            {
                foundIncludes = true;
                var includePath = match.Groups[1].Value;
                var fullPath = ResolvePath(includePath, currentDir);

                if (!File.Exists(fullPath))
                {
                    throw new FileNotFoundException(
                        $"Include file not found: {fullPath} (included from {currentFilePath})");
                }

                // Handle circular includes
                if (processedIncludes.Contains(fullPath))
                {
                    // Simply remove the include directive and continue
                    source.ProcessedContent = content.Remove(match.Index, match.Length);
                    continue;
                }

                processedIncludes.Add(fullPath);

                // Process the included file
                var includeSource = File.ReadAllText(fullPath);
                var includeProcessed = CreateProcessedSource(fullPath, includeSource);

                // Process nested includes first
                includeProcessed = StripComments(includeProcessed);
                includeProcessed = LoadIncludes(includeProcessed, fullPath);

                // Remove the include directive and insert processed content
                source.ProcessedContent = content.Remove(match.Index, match.Length);
                source.InsertContent(match.Index, includeProcessed);
            }
        } while (foundIncludes);

        return source;
    }

    private ProcessedSource ProcessDirectives(ProcessedSource source)
    {
        var lines = source.ProcessedContent.Split('\n');
        var processedLines = new List<string>();
        var newMapping = new List<LineInfo>();

        for (int i = 0; i < lines.Length; i++)
        {
            string trimmedLine = lines[i].Trim();

            // Skip version and extension directives as they're handled separately
            if (!trimmedLine.StartsWith("#version") && !trimmedLine.StartsWith("#extension"))
            {
                processedLines.Add(lines[i]);
                if (i < source.LineMapping.Count)
                {
                    newMapping.Add(source.LineMapping[i]);
                }
            }
        }

        return new ProcessedSource(string.Join("\n", processedLines), newMapping);
    }

    private string ResolvePath(string includePath, string currentDir)
    {
        try
        {
            // First try relative to current file
            string relativePath = Path.GetFullPath(Path.Combine(currentDir, includePath));
            if (File.Exists(relativePath))
            {
                return relativePath;
            }

            // Then try relative to root shader directory
            string rootRelativePath = Path.GetFullPath(Path.Combine(rootDirectory, includePath));
            if (File.Exists(rootRelativePath))
            {
                return rootRelativePath;
            }

            // Finally try absolute path
            if (Path.IsPathRooted(includePath) && File.Exists(includePath))
            {
                return includePath;
            }

            throw new FileNotFoundException($"Could not resolve include path: {includePath}");
        }
        catch (Exception ex)
        {
            throw new ArgumentException($"Invalid include path: {includePath}", ex);
        }
    }

    private List<ShaderError> ParseShaderErrors(string infoLog)
    {
        var errors = new List<ShaderError>();
        var errorLines = infoLog.Split('\n', StringSplitOptions.RemoveEmptyEntries);

        foreach (var errorLine in errorLines)
        {
            // Match both line number formats: (123) and 0:123
// Match both line number formats: (123) and 0:123
            var match = Regex.Match(errorLine, @"(?:\((\d+)\)|0:(\d+))\s*:\s*(.+)");
            if (match.Success)
            {
                string lineNumberStr = match.Groups[1].Success ? match.Groups[1].Value : match.Groups[2].Value;
                int processedLineNumber = int.Parse(lineNumberStr) - 1; // -1 because GL line numbers are 1-based
                if (processedLineNumber < processedSource.LineMapping.Count)
                {
                    var lineInfo = processedSource.LineMapping[processedLineNumber];
                    errors.Add(new ShaderError
                    {
                        FilePath = lineInfo.FilePath,
                        LineNumber = lineInfo.OriginalLineNumber,
                        LineContent = lineInfo.OriginalContent,
                        ErrorMessage = match.Groups[3].Value.Trim()
                    });
                }
            }
        }

        return errors;
    }

    private void CompileShader()
    {
        uint newHandle = 0;
        try
        {
            // Create new shader first
            newHandle = gl.CreateShader((GLEnum)shaderType);
            var error = gl.GetError();

            if (newHandle == 0 || error != GLEnum.NoError)
            {
                Console.WriteLine($"Failed to create shader! Handle: {newHandle}, Error: {error}");
                throw new InvalidOperationException($"Failed to create shader: {error}");
            }

            // Set source and compile with new handle
            gl.ShaderSource(newHandle, processedSource.ProcessedContent);
            gl.CompileShader(newHandle);

            gl.GetShader(newHandle, GLEnum.CompileStatus, out int status);
            if (status == 0)
            {
                var infoLog = gl.GetShaderInfoLog(newHandle);
                Console.WriteLine($"Shader compilation failed:\n{infoLog}");
                var errors = ParseShaderErrors(infoLog);
                OnCompilationError?.Invoke(errors);
                throw new ShaderCompilationException(errors);
            }

            // If compilation succeeded, clean up old handle
            if (handle != 0)
            {
                gl.DeleteShader(handle);
            }

            // Only assign new handle after successful compilation
            handle = newHandle;
            newHandle = 0; // Prevent cleanup in finally block

            OnRecompiled?.Invoke();
        }
        catch (Exception)
        {
            // If anything went wrong, clean up the new handle
            if (newHandle != 0)
            {
                Console.WriteLine($"Cleaning up failed shader handle: {newHandle}");
                gl.DeleteShader(newHandle);
            }
            throw;
        }
    }

    private void SetupFileWatchers()
    {
        // Watch main shader file
        AddFileWatcher(mainShaderPath);

        // Watch all included files
        foreach (var includePath in processedIncludes)
        {
            AddFileWatcher(includePath);
        }
    }

    private void AddFileWatcher(string filePath)
    {
        if (fileWatchers.ContainsKey(filePath))
            return;

        var directory = Path.GetDirectoryName(filePath);
        var filename = Path.GetFileName(filePath);

        if (directory == null)
            return;

        var watcher = new FileSystemWatcher(directory)
        {
            Filter = filename,
            EnableRaisingEvents = true,
            NotifyFilter = NotifyFilters.LastWrite | NotifyFilters.CreationTime
        };

        watcher.Changed += OnShaderFileChanged;

        fileWatchers[filePath] = watcher;
    }

    private void OnShaderFileChanged(object sender, FileSystemEventArgs e)
    {
        Console.WriteLine($"File changed: {e.FullPath}");
        lock (recompileLock)
        {
            needsRecompile = true;
        }
    }

    public void Update()
    {
        bool shouldRecompile = false;
        lock (recompileLock)
        {
            shouldRecompile = needsRecompile;
            needsRecompile = false;
        }

        if (shouldRecompile)
        {
            try
            {
                Console.WriteLine("Recompiling shader...");
                var oldIncludes = new HashSet<string>(processedIncludes);

                CompileInitial();

                // Remove watchers for files that are no longer included
                var removedIncludes = oldIncludes.Except(processedIncludes).ToList();
                foreach (var removedInclude in removedIncludes)
                {
                    if (fileWatchers.TryGetValue(removedInclude, out var watcher))
                    {
                        watcher.Dispose();
                        fileWatchers.Remove(removedInclude);
                    }
                }

                // Add watchers for new includes
                foreach (var newInclude in processedIncludes)
                {
                    AddFileWatcher(newInclude);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error during shader recompilation: {ex}");
            }
        }
    }

    public void Dispose()
    {
        Console.WriteLine($"Disposing shader loader: {mainShaderPath} with handle {handle}");

        // Force synchronization before disposal
        gl.Finish();

        if (handle != 0)
        {
            gl.DeleteShader(handle);
            handle = 0;
        }

        foreach (var watcher in fileWatchers.Values)
        {
            watcher.Dispose();
        }
        fileWatchers.Clear();

        GC.SuppressFinalize(this);
    }
}

