using System;
using System.IO;
using System.Text.Json;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Threading;
using System.Linq;

[StructLayout(LayoutKind.Explicit)]
public struct Material
{
    // Original Material struct definition remains the same
    [FieldOffset(0)] public Vector4 color1;
    [FieldOffset(16)] public Vector4 color2;
    [FieldOffset(32)] public Vector4 color3;
    [FieldOffset(48)] public Vector4 color4;
    [FieldOffset(64)] public float noiseSize;
    [FieldOffset(68)] public float noiseStrength;
    [FieldOffset(72)] public uint texture1Id;
    [FieldOffset(76)] public uint texture2Id;
    [FieldOffset(80)] public float textureBlend;
    [FieldOffset(84)] public bool blendWithNoise;
    [FieldOffset(88)] public bool isPowder;
    [FieldOffset(92)] public bool isLiquid;
    [FieldOffset(96)] public bool isCollidable;
    [FieldOffset(100)] public float roughness;
    [FieldOffset(104)] public float shininess;
    [FieldOffset(108)] public float specularStrength;
    [FieldOffset(112)] public float reflectivity;
    [FieldOffset(116)] public float transparency;
    [FieldOffset(120)] public float refractiveIndex;
    [FieldOffset(124)] public float emission;
}

public class Materials : IDisposable
{
    private static readonly object _lock = new object();
    public static Material[] materials;
    private static Dictionary<string, int> _nameToId;
    private static Dictionary<int, string> _idToName;

    private readonly FileSystemWatcher _watcher;
    private readonly string _materialsDirectory;
    private readonly Timer _reloadTimer;
    private volatile bool _pendingReload;

    public static event Action OnMaterialsReloaded;

    private static Materials _instance;
    public static Materials Instance => _instance ??= new Materials();

    public static void Initialize()
    {
        _instance ??= new Materials();
    }

    private Materials()
    {
        // Set up materials directory path
        _materialsDirectory = "./materials/";

        // Ensure materials directory exists
        Directory.CreateDirectory(_materialsDirectory);

        // Initialize watcher for the entire materials directory
        _watcher = new FileSystemWatcher(_materialsDirectory);
        _watcher.Filter = "*.json";
        _watcher.NotifyFilter = NotifyFilters.LastWrite | NotifyFilters.CreationTime | NotifyFilters.FileName;
        _watcher.Changed += OnFileChanged;
        _watcher.Created += OnFileChanged;
        _watcher.Deleted += OnFileChanged;
        _watcher.Renamed += OnFileRenamed;
        _watcher.EnableRaisingEvents = true;

        // Initialize reload timer (debounce mechanism)
        _reloadTimer = new Timer(_ => ReloadMaterials(), null, Timeout.Infinite, Timeout.Infinite);

        // Initial load
        ReloadMaterials();
    }

    private void OnFileChanged(object sender, FileSystemEventArgs e)
    {
        TriggerReload();
    }

    private void OnFileRenamed(object sender, RenamedEventArgs e)
    {
        TriggerReload();
    }

    private void TriggerReload()
    {
        if (!_pendingReload)
        {
            _pendingReload = true;
            _reloadTimer.Change(100, Timeout.Infinite); // Debounce for 100ms
        }
    }

    private void ReloadMaterials()
    {
        try
        {
            var materialFiles = Directory.GetFiles(_materialsDirectory, "*.json")
                                      .OrderBy(f => f) // Ensure consistent ordering
                                      .ToList();

            var newMaterials = new List<Material>();
            var newNameToId = new Dictionary<string, int>();
            var newIdToName = new Dictionary<int, string>();

            for (int i = 0; i < materialFiles.Count; i++)
            {
                try
                {
                    string json = File.ReadAllText(materialFiles[i]);
                    var materialDef = JsonSerializer.Deserialize<MaterialDefinition>(json);

                    if (materialDef == null)
                    {
                        Console.Error.WriteLine($"Error loading material file {materialFiles[i]}: JSON deserialization failed");
                        continue;
                    }

                    if (string.IsNullOrEmpty(materialDef.name))
                    {
                        Console.Error.WriteLine($"Warning: Material file {materialFiles[i]} has no name specified");
                        continue;
                    }

                    newMaterials.Add(materialDef.properties.ConvertDtoToMaterial());
                    newNameToId[materialDef.name] = i;
                    newIdToName[i] = materialDef.name;
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"Error loading material file {materialFiles[i]}: {ex.Message}");
                }
            }

            // Update data structures under lock
            lock (_lock)
            {
                materials = newMaterials.ToArray();
                _nameToId = newNameToId;
                _idToName = newIdToName;
            }

            // Notify listeners
            OnMaterialsReloaded?.Invoke();
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error reloading materials: {ex}");
        }
        finally
        {
            _pendingReload = false;
        }
    }

    // Thread-safe access methods remain the same
    public static Material GetMaterial(int id)
    {
        lock (_lock)
        {
            if (id < 0 || id >= materials.Length)
            {
                return materials[0]; // Return Invalid material
            }
            return materials[id];
        }
    }

    public static string GetMaterialName(int id)
    {
        lock (_lock)
        {
            return _idToName.TryGetValue(id, out string name) ? name : "Unknown";
        }
    }

    public static int GetMaterialId(string name)
    {
        lock (_lock)
        {
            return _nameToId.TryGetValue(name, out int id) ? id : 0;
        }
    }

    public static int GetMaterialCount()
    {
        lock (_lock)
        {
            return materials?.Length ?? 0;
        }
    }

    public void Dispose()
    {
        _watcher?.Dispose();
        _reloadTimer?.Dispose();
    }


    // Helper classes for JSON deserialization
    public class MaterialPropertiesDto
    {
        public Vector4Dto color1 { get; set; }
        public Vector4Dto color2 { get; set; }
        public Vector4Dto color3 { get; set; }
        public Vector4Dto color4 { get; set; }
        public float noiseSize { get; set; }
        public float noiseStrength { get; set; }
        public uint texture1Id { get; set; }
        public uint texture2Id { get; set; }
        public float textureBlend { get; set; }
        public bool blendWithNoise { get; set; }
        public bool isPowder { get; set; }
        public bool isLiquid { get; set; }
        public bool isCollidable { get; set; }
        public float roughness { get; set; }
        public float shininess { get; set; }
        public float specularStrength { get; set; }
        public float reflectivity { get; set; }
        public float transparency { get; set; }
        public float refractiveIndex { get; set; }
        public float emission { get; set; }

        public Material ConvertDtoToMaterial()
        {
            return new Material
            {
                color1 = new Vector4(color1.r, color1.g, color1.b, color1.controlPoint),
                color2 = new Vector4(color2.r, color2.g, color2.b, color2.controlPoint),
                color3 = new Vector4(color3.r, color3.g, color3.b, color3.controlPoint),
                color4 = new Vector4(color4.r, color4.g, color4.b, color4.controlPoint),
                noiseSize = noiseSize,
                noiseStrength = noiseStrength,
                texture1Id = texture1Id,
                texture2Id = texture2Id,
                textureBlend = textureBlend,
                blendWithNoise = blendWithNoise,
                isPowder = isPowder,
                isLiquid = isLiquid,
                isCollidable = isCollidable,
                roughness = roughness,
                shininess = shininess,
                specularStrength = specularStrength,
                reflectivity = reflectivity,
                transparency = transparency,
                refractiveIndex = refractiveIndex,
                emission = emission
            };
        }
    }
    public class Vector4Dto
    {
        public float r { get; set; }
        public float g { get; set; }
        public float b { get; set; }
        public float controlPoint { get; set; }
    }
    public class MaterialDefinition
    {
        public string name { get; set; }
        public MaterialPropertiesDto properties { get; set; }
    }

}