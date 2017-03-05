#include "index.hpp"
#include "vr_hmd.hpp"
#include "vr_renderer.hpp"
#include "static_mesh.hpp"
#include "bullet_debug.hpp"
#include "mpmc_bounded_queue.hpp"
#include "spsc_queue.hpp"
#include "mpsc_queue.hpp"
#include "spsc_bounded_queue.hpp"

using namespace avl;

float4x4 make_directional_light_view_proj(const uniforms::directional_light & light, const float3 & eyePoint)
{
	const Pose p = look_at_pose(eyePoint, eyePoint + -light.direction);
	const float halfSize = light.size * 0.5f;
	return mul(make_orthographic_matrix(-halfSize, halfSize, -halfSize, halfSize, -halfSize, halfSize), make_view_matrix_from_pose(p));
}

float4x4 make_spot_light_view_proj(const uniforms::spot_light & light)
{
	const Pose p = look_at_pose(light.position, light.position + light.direction);
	return mul(make_perspective_matrix(to_radians(light.cutoff * 2.0f), 1.0f, 0.1f, 1000.f), make_view_matrix_from_pose(p));
}

struct ScreenViewport
{
	float2 bmin, bmax;
	GLuint texture;
};

// MotionControllerVR wraps BulletObjectVR and is responsible for creating a controlled physically-activating 
// object, and keeping the physics engine aware of the latest user-controlled pose.
class MotionControllerVR
{
	Pose latestPose;

	void update_physics(const float dt, BulletEngineVR * engine)
	{
		physicsObject->body->clearForces();
		physicsObject->body->setWorldTransform(to_bt(latestPose.matrix()));
	}

public:

	std::shared_ptr<BulletEngineVR> engine;
	const OpenVR_Controller & ctrl;
	std::shared_ptr<OpenVR_Controller::ControllerRenderData> renderData;

	btCollisionShape * controllerShape{ nullptr };
	BulletObjectVR * physicsObject{ nullptr };

	MotionControllerVR(std::shared_ptr<BulletEngineVR> engine, const OpenVR_Controller & ctrl, std::shared_ptr<OpenVR_Controller::ControllerRenderData> renderData)
		: engine(engine), ctrl(ctrl), renderData(renderData)
	{

		engine->add_task([=](float time, BulletEngineVR * engine)
		{
			this->update_physics(time, engine);
		});

		controllerShape = new btBoxShape(btVector3(0.096, 0.096, 0.0123)); // fixme to use renderData

		physicsObject = new BulletObjectVR(new btDefaultMotionState(), controllerShape, engine->get_world(), 0.5f);

		physicsObject->body->setFriction(2.f);
		physicsObject->body->setRestitution(0.75);
		physicsObject->body->setGravity(btVector3(0, 0, 0));
		physicsObject->body->setActivationState(DISABLE_DEACTIVATION);

		engine->add_object(physicsObject);
	}

	void update_controller_pose(const Pose & p)
	{
		latestPose = p;
	}

};

struct Scene
{
	RenderableGrid grid {0.25f, 24, 24 };

	std::unique_ptr<MotionControllerVR> leftController;
	std::unique_ptr<MotionControllerVR> rightController;

	std::vector<std::shared_ptr<BulletObjectVR>> physicsObjects;

	std::vector<StaticMesh> models;
	std::vector<StaticMesh> controllers;

	std::map<std::string, std::shared_ptr<Material>> namedMaterialList;

	std::vector<Renderable *> gather()
	{
		std::vector<Renderable *> objectList;
		for (auto & model : models) objectList.push_back(&model);
		for (auto & ctrlr : controllers) objectList.push_back(&ctrlr);
		return objectList;
	}
};

struct VirtualRealityApp : public GLFWApp
{
	SPSCBoundedQueue<float3> numQueue;

	std::unique_ptr<VR_Renderer> renderer;
	std::unique_ptr<OpenVR_HMD> hmd;

	GlCamera debugCam;
	FlyCameraController cameraController;

	ShaderMonitor shaderMonitor = { "../assets/" };
	
	std::vector<ScreenViewport> viewports;
	Scene scene;

	std::shared_ptr<BulletEngineVR> physicsEngine;
	std::unique_ptr<PhysicsDebugRenderer> physicsDebugRenderer;

	VirtualRealityApp();
	~VirtualRealityApp();

	void setup_physics();

	void setup_scene();

	void on_window_resize(int2 size) override;

	void on_input(const InputEvent & event) override;

	void on_update(const UpdateEvent & e) override;

	void on_draw() override;
};