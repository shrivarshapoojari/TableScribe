import React, { useState } from 'react';

const TableScribe = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [imagePreview, setImagePreview] = useState(null);

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        setSelectedFile(file);

        // Preview image
        const reader = new FileReader();
        reader.onloadend = () => {
            setImagePreview(reader.result);
        };
        if (file) {
            reader.readAsDataURL(file);
        }
    };

    const handleUpload = async () => {
        if (!selectedFile) {
            alert('Please select a file.');
            return;
        }

        // Create FormData object
        const formData = new FormData();
        formData.append('image', selectedFile);

        // Replace with your backend endpoint
        const url = 'https://example.com/upload'; // Replace with your actual backend URL

        try {
            const response = await fetch(url, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error('Upload failed');
            }

            alert('Upload successful!');
            // Optionally handle response data here
        } catch (error) {
            console.error('Error uploading image:', error);
            alert('Upload failed. Please try again.');
        }
    };

    const handleRemoveImage = () => {
        setSelectedFile(null);
        setImagePreview(null);
    };

    return (
        <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center">
            <div className="max-w-lg w-full bg-white p-8 rounded-lg shadow-md">
                <h1 className="text-3xl font-bold mb-6 text-center">Table Scribe</h1>
                <div className="relative mb-6">
                    {imagePreview ? (
                        <div className="mb-4 flex items-center justify-center">
                            <img src={imagePreview} alt="Uploaded" className="rounded-lg max-h-60 mx-auto mb-4" />
                        </div>
                    ) : (
                        <div className="border-dashed border-2 border-gray-400 p-6 rounded-lg text-center">
                            <span className="text-lg text-gray-500">No image selected</span>
                        </div>
                    )}
                    <div className="flex items-center justify-center space-x-4">
                        <label htmlFor="file-upload" className="block mt-4">
                            <input id="file-upload" type="file" className="hidden" onChange={handleFileChange} />
                            <button
                                className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
                                onClick={() => document.getElementById('file-upload').click()}
                            >
                                {imagePreview ? 'Change Image' : 'Select Image'}
                            </button>
                        </label>
                        {imagePreview && (
                            <button
                                className="bg-red-500 hover:bg-red-600 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline mt-4"
                                onClick={handleRemoveImage}
                            >
                                Remove Image
                            </button>
                        )}
                    </div>
                </div>
                {imagePreview && (
                    <div className="text-center">
                        <button
                            className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
                            onClick={handleUpload}
                        >
                            Upload
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
};

export default TableScribe;
