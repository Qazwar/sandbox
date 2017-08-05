#include "index.hpp"
#include "holo-scan-effect.hpp"
#include "terrain-scan-effect.hpp"
#include "fade-to-skybox.hpp"
#include "cheap-subsurface-scattering.hpp"
#include "area-light-ltc.hpp"
#include "lab-teleportation-sphere.hpp"
#include "triplanar-terrain.hpp"

struct shader_workbench : public GLFWApp
{
    GlCamera cam;
    FlyCameraController flycam;
    ShaderMonitor shaderMonitor{ "../assets/" };
    std::unique_ptr<gui::ImGuiManager> igm;
    GlGpuTimer gpuTimer;
    float elapsedTime{ 0 };
    float triangleScale{ 0.1f };

    std::shared_ptr<GlShader> holoScanShader, wireframeShader, basicShader;
    GlMesh terrainMesh;
    GlMesh sofaMesh, headsetMesh, frustumMesh;

    shader_workbench();
    ~shader_workbench();

    virtual void on_window_resize(int2 size) override;
    virtual void on_input(const InputEvent & event) override;
    virtual void on_update(const UpdateEvent & e) override;
    virtual void on_draw() override;
};