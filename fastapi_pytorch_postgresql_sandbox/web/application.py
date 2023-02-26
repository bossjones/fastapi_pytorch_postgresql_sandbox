"""web.application"""
# sourcery skip: avoid-global-variables
# pylint: disable=no-name-in-module

from importlib import metadata
from pathlib import Path

from fastapi import FastAPI, UploadFile
from fastapi.responses import UJSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from fastapi_pytorch_postgresql_sandbox.logging import configure_logging
from fastapi_pytorch_postgresql_sandbox.web.api.router import api_router

# from fastapi_pytorch_postgresql_sandbox.web.driver import RayEntryPoint
from fastapi_pytorch_postgresql_sandbox.web.lifetime import (
    register_shutdown_event,
    register_startup_event,
)

APP_ROOT = Path(__file__).parent.parent
# DISABLED: # ray.init(address="auto")
# DISABLED: # ray_http_options = {"http_options": {"host": "0.0.0.0", "port": 7779}}
# DISABLED: ray.init()
# DISABLED: # serve.start(detached=True, **ray_http_options)
# DISABLED: serve.start(detached=True)
# DISABLED:
# DISABLED: # ray.init(address=cluster.address, log_to_driver=False, dashboard_host="0.0.0.0")


class ImageDTO(BaseModel):
    """Data Transfer Object(DTO) for uploaded images."""

    # data: Any
    file: UploadFile


def get_app() -> FastAPI:
    """
    Get FastAPI application.

    This is the main constructor of an application.

    :return: application.
    """
    configure_logging()
    app = FastAPI(
        title="fastapi_pytorch_postgresql_sandbox",
        version=metadata.version("fastapi_pytorch_postgresql_sandbox"),
        docs_url=None,
        redoc_url=None,
        openapi_url="/api/openapi.json",
        default_response_class=UJSONResponse,
    )

    # Adds startup and shutdown events.
    register_startup_event(app)
    register_shutdown_event(app)

    # Main router for the API.
    app.include_router(router=api_router, prefix="/api")
    # Adds static directory.
    # This directory is used to access swagger files.
    app.mount(
        "/static",
        StaticFiles(directory=APP_ROOT / "static"),
        name="static",
    )

    # # New FastAPI HTTP Deployments running on uvicorn
    # # NOTE: https://discuss.ray.io/t/new-fastapi-http-deployments-running-on-uvicorn/3435/4
    # # SOURCE: https://docs.ray.io/en/latest/serve/api/rest_api.html#serve-rest-api-config-schema
    # @serve.deployment(route_prefix="/ml")
    # @serve.ingress(app)
    # class ScreenShotFastAPIDeployment:
    #     def __init__(self):
    #         # Load model
    #         self.container = ImageClassifier()
    #         self.container.load_model()
    #         self.model = self.container.model

    #     def classify(self, image_payload_bytes: Any):
    #         """Take an image and classify which type of screenshot it is

    #         Args:
    #             image_payload_bytes (BytesIO): _description_

    #         Returns:
    #             _type_: _description_
    #         """

    #         # 4. Create empty dictionary to store prediction information for each sample
    #         pred_dict = {}

    #         # 6. Start the prediction timer
    #         start_time = timer()

    #         pil_image = Image.open(BytesIO(image_payload_bytes))  # orig

    #         # 7. Open image path
    #         # img = Image.open(path)
    #         img = convert_pil_image_to_rgb_channels(pil_image)

    #         # pil_images = [pil_image]  # batch size is one
    #         pil_images = [img]  # batch size is one
    #         input_tensor = torch.cat(
    #             [self.container.auto_transforms(i).unsqueeze(dim=0) for i in pil_images]
    #         )  # orig

    #         transformed_image = input_tensor

    #         with torch.inference_mode():  # orig
    #             pred_logit = self.model(
    #                 transformed_image.to(self.container.device)
    #             )  # perform inference on target sample
    #             pred_prob = torch.softmax(
    #                 pred_logit, dim=1
    #             )  # turn logits into prediction probabilities
    #             pred_label = torch.argmax(
    #                 pred_prob, dim=1
    #             )  # turn prediction probabilities into prediction label
    #             pred_class = self.container.class_names[
    #                 pred_label.cpu()
    #             ]  # hardcode prediction class to be on CPU

    #             # 11. Make sure things in the dictionary are on CPU (required for inspecting predictions later on)
    #             pred_dict["pred_prob"] = round(
    #                 pred_prob.unsqueeze(0).max().cpu().item(), 4
    #             )
    #             pred_dict["pred_class"] = pred_class

    #             # 12. End the timer and calculate time per pred
    #             end_time = timer()
    #             pred_dict["time_for_pred"] = round(end_time - start_time, 4)

    #             # with torch.no_grad():  # orig
    #             # output_tensor = self.model(input_tensor)  # orig

    #         # res = {"class_index": int(torch.argmax(output_tensor[0]))}  # orig
    #         # print(res)

    #         res = [pred_dict]
    #         print(res)
    #         return res

    #     @app.get("/")
    #     def get(self):
    #         return "Welcome to the ScreenNet model server."

    #     @app.post("/classify_image")
    #     async def classify_image(self, file: UploadFile = File(...)):
    #         image_bytes = await file.read()
    #         return self.classify(image_bytes)

    # ScreenShotFastAPIDeployment.deploy()  # pylint: disable=no-member

    return app
